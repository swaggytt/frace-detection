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


def main():
    file_upload = st.file_uploader(
        "Choose the file", type=['jpg', 'png', 'jpeg'])
    if file_upload is not None:
        image = Image.open(file_upload)
        new_img = np.array(image.convert('RGB'))
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        st.image(new_img)
        st.image(har(image))
        

st.header('Image class predictator')
main()
