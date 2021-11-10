import numpy as np
import streamlit as st
from PIL import Image
from deepface import DeepFace
import torch
from torchvision import models, transforms
import torch.nn as nn

WEIGHT_PATH = "./resnet-18_weight_v11.pth"

# @st.cache
def deepface(image_file):
    detected_face = DeepFace.detectFace(image_file, detector_backend='dlib', enforce_detection=False)
    return detected_face

@st.cache
def transforms_picture(face_img):
    face_img = np.rot90(np.flip(face_img), 2)
    res = face_img*255
    im = Image.fromarray(np.uint8(res)).convert('RGB')
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    image = data_transforms(im)
    image = image.unsqueeze(0)
    return image

@st.cache
def create_model(device):
    # load pretrain
    model = models.resnet18()

    # modified model layer
    num_ftrs = model.fc.in_features
    features = list(model.fc.children())[:-1]  # Remove last layer
    features.extend([nn.Linear(num_ftrs, 256), nn.ReLU(), nn.Dropout(p=0.8), nn.Linear(256, 3), nn.Softmax(dim=1)])  # Add our layer with 3 outputs
    model.fc = nn.Sequential(*features)  # Replace the model classifier

    # load weight
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))

    return model 

def predict_race(image, model, device):
    model.eval()
    image = image.to(device)
    pred = model(image)
    return pred
    

def main():
    st.header('Image race predictator')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_upload = st.file_uploader("Choose the file", type=['jpg', 'png', 'jpeg', 'jfif'])
    col1, col2 = st.columns([3, 1])
    if file_upload is not None:

        with st.spinner('Processing...'):
            # add image 
            image = Image.open(file_upload)

            # preprocess
            new_img = np.array(image)
            col1.image(new_img, width=500)

            # face detection with deepface
            image = deepface(new_img)
            col2.image(image, channels="BGR", width=200)

            # transform picture
            transforms_im = transforms_picture(image)

            # init model
            model = create_model(device)
            
            # predict face
            pred = predict_race(transforms_im, model, device)
            _, pred_idx = torch.max(pred.data, 1)
        if pred_idx == 0:
            col2.subheader("Predict : Chinese")
        elif pred_idx == 1:
            col2.subheader("Predict : Malaysian")
        else:
            col2.subheader("Predict : Thai")
        col2.subheader(f"Probability:")
        col2.write(f"Chainese {pred.data[0][0]*100:.2f}%")
        col2.write(f"Malaysian {pred.data[0][1]*100:.2f}%")
        col2.write(f"Thai {pred.data[0][2]*100:.2f}%")
        st.success("Success!")
        # print(f"Probability: \nChainese {pred.data[0][0]*100:.2f}%, \nMalaysian {pred.data[0][1]*100:.2f}%, \nThai {pred.data[0][2]*100:.2f}%")
        


main()
