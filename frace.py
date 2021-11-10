from keras.backend import dropout
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import os
import h5py
import cv2
from PIL import Image
# from cv2 import imread
# from cv2 import waitKey
# from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle
from deepface import DeepFace
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn


@st.cache
def har(image_file):
    # load the pre-trained model
    classifier = CascadeClassifier('./haarcascade_frontalface_default.xml')
    pixels = np.array(image_file.convert('RGB'))
    bboxes = classifier.detectMultiScale(pixels)
    for box in bboxes:
        # extract
        x, y, width, height = box
        x2, y2 = x + width, y + height
        # draw a rectangle over the pixels
        rectangle(pixels, (x, y), (x2, y2), (255, 0, 0), 1)
    return pixels


@st.cache
def deepface(image_file):
    detected_face = DeepFace.detectFace(
        image_file, detector_backend='dlib', enforce_detection=False)
    return np.rot90(np.flip(detected_face), 2)


def main():
    file_upload = st.file_uploader(
        "Choose the file", type=['jpg', 'png', 'jpeg'])
    if file_upload is not None:
        image = Image.open(file_upload)
        new_img = np.array(image.convert('RGB'))
        st.image(new_img)
        face_img = deepface(new_img)
        st.image(face_img)
        res = face_img*255
        im = Image.fromarray(np.uint8(res)).convert('RGB')
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        image = data_transforms(im)
        image = image.unsqueeze(0)
        model = models.resnet18()

        # num_ftrs = model.fc.in_features
        # features = list(model.fc.children())[:-1]  # Remove last layer
        # features.extend([nn.Flatten(), nn.Linear(num_ftrs, 256), nn.ReLU(), nn.Dropout(
        #     p=0.8), nn.Linear(256, 3), nn.Softmax(dim=1)])  # Add our layer with 3 outputs
        # # features = [nn.Linear(num_ftrs, 256),nn.Dropout(p=0.8),nn.Linear(256,len(class_names)) ]  # Add our layer with 3 outputs
        # model.fc = nn.Sequential(*features)  # Replace the model classifier

        num_ftrs = model.fc.in_features
        features = list(model.fc.children())[:-1]  # Remove last layer
        features.extend([nn.Linear(num_ftrs, 256), nn.ReLU(), nn.Dropout(p=0.8), nn.Linear(
            256, 3), nn.Softmax(dim=1)])  # Add our layer with 3 outputs  #
        model.fc = nn.Sequential(*features)  # Replace the model classifier

        pre_layer = list(model.children())
        # change first layer dim to 1,64 for grayscale
        pre_layer.insert(0, nn.Conv2d(1, 3, kernel_size=3,
                         stride=1, padding=1, bias=False))
        model = nn.Sequential(*pre_layer)

        weight = torch.load("./resnet-18_weight_v11.pth",
                            map_location='cpu')  # init weight

        model.load_state_dict(weight)

        model.eval()

        pred = model(image)

        print(pred)

        _, pred = torch.max(pred.data, 1)
        print(pred)

        if pred == 0:
            st.write("Predict : chinese")
        elif pred == 1:
            st.write("Predict : malaysia")
        else:
            st.write("Predict : thai")
        st.write(res)


st.header('Image class predictator')
main()
