# Virtual Makeup Try-On

This project is for  IITI-Summer-of-Code 2024 (IITI-SoC'24) in AI/ML division and secured🥈Second position.


## Overview
Welcome to Virtual-Makeup-Try-on  repository!! This project uses  OpenCV for capturing and processing video input, and MediaPipe for precise real-time facial landmark detection to apply virtual makeup accurately.This project is as well focused on utilising the key features of Generative Adversarial Networks (GANs) to create realistic and accurate simulations of how makeup products will look on a user's face,Using This model is pre-trained on the Beauty-GAN dataset for instant application of realistic makeup.This README file serves as a comprehensive guide to understanding the project, setting up the environment, running the model, and contributing to the development
## Introduction
In this era of online shopping , the masses virtually wanna try on their makeup before buying the products . We made a seamless software to save time and avoid the hassle of in-store trials. We let the user to experiment with various products including lipsticks, eyeshadows, toner, and more, without any commitment. Our software is easy to navigate and fun to use ,while providing the users with a wide range of colours and different shade pallets to choose from.Simply either  upload a photo or use your device’s camera to start trying on makeup instantly with the bonus feature of model try on equipped with before and after images.We utilised tech such as OpenCV is utilized for efficient image processing and facial feature detection. MediaPipe enhances this by providing real-time face tracking and landmark detection.Our project also leverage BeautyGAN to generate realistic and high-quality makeup transformations.

## Resources/Links:
Mediapipe Face Mesh Concepts : To access the face landmark detection code .
mediapipe/docs/solutions/face_mesh.md at master · google-ai-edge/mediapipe (github.com) 

Mediapipe Face Landmarks Points 
\tfjs-models/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts at 838611c02f51159afdd77469ce67f0e26b7bbb23 · tensorflow/tfjs-models (github.com) 

Dlib Face Landmarks 
Facial Landmarks for Face Recognition with Dlib - Sefik Ilkin Serengil (sefiks.com) 

(blog): 
https://pavankunchalapk.medium.com/virtual-makeu p-app-using-streamlit-and-opencv-b1271b8e2d01 
beautyGAN: 

GitHub - Honlan/BeautyGAN: transfer the makeup style of a reference face image to a non-makeup face 
Beauty Gan (kaggle.com) 

(blog): 
https://medium.com/@sriskandaryan/facial-makeup-t ransfer-beautygan-d99389e1aae4 

Open cv:
https://opencv.org/

Streamlit:
https://streamlit.io/
## Key Features
* Utilizes a GAN-bases approach to try on makeup
* The user's image is processed using OpenCV to ensure it is in the correct format and quality for further analysis.Here OpenCV uses techniques like Haar cascades and deep learning-based methods to detect facial features (eyes, nose, mouth, etc.) in images, providing essential landmarks for accurate makeup application.
* Utilising the Beauty -GAN  to generate realistic makeup effects . GAN utilises two neural networks one being discriminator and other being generator working symbiotically creating lifelike makeup transformations.
* Utilising the Media pipe detecting and tracking facial landmarks and simultaneously integrating BeautyGAN using these landmarks to transfer makeup styles from reference images, generating lifelike and naturally blended results.
* The final image, with the applied makeup, is displayed in the Streamlit web application. Users can upload new images, select different makeup styles, and see the results in real-time.


## Installation

1)Clone this GitHub repository: git clone
```bash
git clone https://github.com/garvitkumbhat619/Virtual-Makeup-Try-on.git
cd Virtual-Makeup-Try-on

```
As an easier alternative, you can download the zip file of the repo and extract it. 
2) Install the required dependencies:
```bash
pip install -r requirements.txt
```


## Troubleshooting
* Issue: Application crashes or fails to start. Solution: Ensure all dependencies are installed and check for any missing files or configuration errors.

## Contributions 
1. Garvit Kumbhat (Chemical Engineering , IIT Indore)
2. Abhiraj Kumar (Mathematics and Computing ,IIT Indore)
3. Bhumika Aggarwal (Metallurgical and Material Science , IIT Indore)
4. Harsh Bhati (Metallurgical and Material Science ,IIT Indore)
