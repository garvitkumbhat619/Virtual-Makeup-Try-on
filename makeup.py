import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from skimage.filters import gaussian
from PIL import Image
import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf

st.sidebar.image("_079d4729-3b9b-45ba-97f9-f4ea8dd88a83.jpg",width=100) # path of _079d4729-3b9b-45ba-97f9-f4ea8dd88a83.jpg
st.title("VIRTUAL MAKEUP TRY-ON")
status1=st.sidebar.selectbox("SELECT AN OPTION:",('LIVE MAKEUP TRY-ON','MAKEUP ON PHOTO','MAKEUP TRANSFER'))

# Function to create mask
def create_mask(points,landmarks,image_width,image_height,image):
    points = [(int(landmarks[p].x * image_width), int(landmarks[p].y * image_height)) for p in points]
    points = np.array(points, dtype=np.int32)
    mask = np.zeros(image.shape, dtype=np.uint8)
    return mask,points

if status1=='LIVE MAKEUP TRY-ON':
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
###################################################################################################################################
                                      # ALL FUNCTIONS DEFINED
###################################################################################################################################                            
    concealer_n=1
# Function to apply lipliner 
    def apply_lipliner(image, landmarks, image_width, image_height, color,):
        lips_points = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,61
            ]
        lips_points = np.array(
        [(int(landmarks[point].x * image_width), int(landmarks[point].y * image_height)) for point in lips_points],
        np.int32)
        demo=np.zeros(image.shape, dtype=np.uint8)
        cv2.polylines(demo, [lips_points], isClosed=False, color=color, thickness=1)
        demo=cv2.GaussianBlur(demo,(3,3),2)
        image=cv2.addWeighted(image,1,demo,0.5,0)
        return image

# Function to convert hex color to BGR format
    def hex_to_bgr(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) for i in (4, 2, 0))

# Function to apply lipstick with improved blending
    def apply_lipstick(image, landmarks, image_width, image_height, color):
        lips_points = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61,
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78
        ]
        mask,lips_points=create_mask(lips_points,landmarks,image_width,image_height,image)
        cv2.fillPoly(mask, [lips_points], color)
        mask=cv2.GaussianBlur(mask,(7,7),0.5)
        image = cv2.addWeighted(image, 1, mask, 0.2, 0)
        return image

# Function to apply undereye 
    def apply_undereye(image, landmarks, image_width, image_height, color):
        left_eye_bottom = [133, 155, 154, 153, 145, 144, 163, 7,33]
        right_eye_bottom = [362, 382, 381, 380, 374, 373, 390, 249,263]

        left_eye_bottom_points = np.array(
            [(int(landmarks[point].x * image_width), int(landmarks[point].y * image_height)) for point in left_eye_bottom],
            np.int32)
        right_eye_bottom_points = np.array(
            [(int(landmarks[point].x * image_width), int(landmarks[point].y * image_height)) for point in right_eye_bottom],
            np.int32)
        cv2.polylines(image, [left_eye_bottom_points], isClosed=False, color=color, thickness=1)
        cv2.polylines(image, [right_eye_bottom_points], isClosed=False, color=color, thickness=1)
        return image

# Function to apply blush with improved blending
    def apply_blush(image, landmarks, image_width, image_height, color):
       blush1_points = [
            280,411,371,352,345
        ]
       mask,blush1_points=create_mask(blush1_points,landmarks,image_width,image_height,image)
       cv2.fillPoly(mask, [blush1_points], color)
       mask = cv2.GaussianBlur(mask, (35, 35), 20)
       image = cv2.addWeighted(image,1, mask, 0.2, 0)
       blush2_points = [
           187,147,137,116,50
           ]
       mask,blush2_points=create_mask(blush2_points,landmarks,image_width,image_height,image)
       cv2.fillPoly(mask, [blush2_points], color)
       mask = cv2.GaussianBlur(mask, (41,41), 24)
       image = cv2.addWeighted(image,1, mask, 0.2, 0)
       return image
    
# Function to apply concealer  
    def apply_concealer(image, landmarks, image_width, image_height, color,concealer_n):
        concealer_points = [
            83,18,313,421,428,396,175,171,208,201
        ]
        if concealer_n==2:
            concealer_points = [
                1,45,51,3,196,122,193,108,151,337,417,351,419,248,281,275
            ]
        if concealer_n==3:
            concealer_points = [
                412,277,266,280,345,454,356,249,390,373,374,380,381,382,362
            ]
            mask,concealer_points=create_mask(concealer_points,landmarks,image_width,image_height,image)
            cv2.fillPoly(mask, [concealer_points], color)
            mask = cv2.GaussianBlur(mask, (35, 35), 10)
            image = cv2.addWeighted(image, 1, mask, 0.15, 0)
            concealer_points=[133, 155, 154, 153, 145, 144, 163, 7,33,34,227,116,50,36,47,188]
        mask,concealer_points=create_mask(concealer_points,landmarks,image_width,image_height,image)
        cv2.fillPoly(mask, [concealer_points], color)
        mask = cv2.GaussianBlur(mask, (35, 35), 10)
        image = cv2.addWeighted(image, 1, mask, 0.15, 0)
        return image

#function to apply eyeshadow
    def apply_eyeshadow(image, landmarks, image_width, image_height,color):
        eyeshadow_points = [
            414,286,258,257,259,467,445,444,443,442,441
            ]
        mask,eyeshadow_points=create_mask(eyeshadow_points,landmarks,image_width,image_height,image)
        cv2.fillPoly(mask, [eyeshadow_points], color)
        mask = cv2.GaussianBlur(mask, (25, 25), 15)
        image = cv2.addWeighted(image,1, mask, 0.6, 0)
        eyeshadow_points = [
            190,56,28,27,29,30,225,224,223,222,221
            ]
        mask,eyeshadow_points=create_mask(eyeshadow_points,landmarks,image_width,image_height,image)
        cv2.fillPoly(mask, [eyeshadow_points], color)
        mask = cv2.GaussianBlur(mask, (25, 25), 15)
        image = cv2.addWeighted(image,1, mask, 0.6, 0)
        return image


# Function to apply eyebrows with improved blending
    def apply_eyebrows(image, landmarks, image_width, image_height, color):
        eyebrow_points = [
            55,107,66,105,63,70,53,52,55
        ]
        mask,eyebrow_points=create_mask(eyebrow_points,landmarks,image_width,image_height,image)
        cv2.fillPoly(mask, [eyebrow_points], color)
        mask = cv2.GaussianBlur(mask, (25, 25), 15)
        image = cv2.addWeighted(image, 1, mask, 0.4, 0)
        eyebrow_points = [
            285,295,282,283,276,293,334,296,336
        ]
        mask,eyebrow_points=create_mask(eyebrow_points,landmarks,image_width,image_height,image)
        cv2.fillPoly(mask, [eyebrow_points], color)
        mask = cv2.GaussianBlur(mask, (25, 25), 15)
        image = cv2.addWeighted(image, 1, mask, 0.4, 0)
        return image

# Function to apply skin toner with improved blending
    def apply_skin_toner(image, landmarks, image_width, image_height, color): 
        skin_points = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        mask,skin_points=create_mask(skin_points,landmarks,image_width,image_height,image)
        cv2.fillPoly(mask, [skin_points], color)
        mask = cv2.GaussianBlur(mask, (11,11), 10)
        return cv2.addWeighted(image, 1, mask,0.08, 0)

#########################################################################################################################################
    st.sidebar.title("Select Makeup Products and Colors")

# Main function
    def main():
        lipstick = st.sidebar.checkbox("Lipstick")
        lipstick_color = st.sidebar.color_picker("Lipstick Color", "#BF0909")
        lipstick_color = hex_to_bgr(lipstick_color)
    
        concealer=st.sidebar.checkbox("Concealer")
        concealer_color=st.sidebar.color_picker("Concealer Color", "#D68821")
        concealer_color = hex_to_bgr(concealer_color)
        if concealer:
            status=st.radio("Choose Place to apply Concealer:",('Chin','Nose','Under Eye'))
            if status=='Chin':
                concealer_n=1
            if status=='Nose':
                concealer_n=2
            if status=='Under Eye':
                concealer_n=3

        undereye = st.sidebar.checkbox("Undereye")
        eye_liner_color = st.sidebar.color_picker("Undereye", "#000000")
        eye_liner_color = hex_to_bgr(eye_liner_color)

        blush = st.sidebar.checkbox("Blush")
        blush_color = st.sidebar.color_picker("Blush Color", "#FF0000")
        blush_color = hex_to_bgr(blush_color)

        lipliner = st.sidebar.checkbox("Lipliner")
        lipliner_color = st.sidebar.color_picker("Lipliner Color", "#874511")
        lipliner_color = hex_to_bgr(lipliner_color)

        eyebrows = st.sidebar.checkbox("Eyebrows")
        eyebrow_color = st.sidebar.color_picker("Eyebrow Color", "#000000")
        eyebrow_color = hex_to_bgr(eyebrow_color)

        toner = st.sidebar.checkbox("Skin Toner")
        toner_color = st.sidebar.color_picker("Skin Toner Color", "#E0AC69")
        toner_color = hex_to_bgr(toner_color)
        
        eyeshadow= st.sidebar.checkbox("Eye Shadow")
        eyeshadow_color= st.sidebar.color_picker("Eye Shadow Color","#FF0000")
        eyeshadow_color= hex_to_bgr(eyeshadow_color)

    # Video capture
        run = st.toggle('Run')
        ba=st.toggle('Before/After')
        FRAME_WINDOW = st.image([])

        camera = cv2.VideoCapture(0)

        while run:
            ret, frame = camera.read()
            orig=frame
            if not ret:
                st.warning("Failed to capture image from the webcam")
                continue

            image_height, image_width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark
                    
                    if toner:
                        frame = apply_skin_toner(frame, landmarks, image_width, image_height, toner_color)
                    if concealer:
                        frame=apply_concealer(frame, landmarks, image_width, image_height, concealer_color,concealer_n)
                    if undereye:
                        frame = apply_undereye(frame, landmarks, image_width, image_height, eye_liner_color)
                    if blush:
                        frame = apply_blush(frame, landmarks, image_width, image_height, blush_color)
                    if lipliner:
                        frame = apply_lipliner(frame, landmarks, image_width, image_height, lipliner_color)
                    if eyebrows:
                        frame = apply_eyebrows(frame, landmarks, image_width, image_height, eyebrow_color)
                    if eyeshadow:
                        frame= apply_eyeshadow(frame,landmarks,image_width,image_height,eyeshadow_color)
                    if lipstick:
                        frame = apply_lipstick(frame, landmarks, image_width, image_height, lipstick_color)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            if ba:
                image_=np.zeros(orig.shape,np.uint8)
                width=image_width
                height=image_height
                w=width//2
                h=height
                smaller_frame=cv2.resize(frame,(0,0),fx=0.5,fy=1)
                smaller_orig=cv2.resize(orig,(0,0),fx=0.5,fy=1)
                image_[:h,:w]=smaller_orig
                image_[:h,w:]=smaller_frame
                FRAME_WINDOW.image(image_)
            else:
                FRAME_WINDOW.image(frame)

        camera.release()

    main()   

if status1=='MAKEUP ON PHOTO': 
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
###################################################################################################################################
                                      # ALL FUNCTIONS DEFINED
###################################################################################################################################                            
    concealer_n=1
## Function to apply lipliner 
    def apply_lipliner(image, landmarks, image_width, image_height, color,):
        lips_points = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,61
            ]
        lips_points = np.array(
        [(int(landmarks[point].x * image_width), int(landmarks[point].y * image_height)) for point in lips_points],
        np.int32)
        demo=np.zeros(image.shape, dtype=np.uint8)
        cv2.polylines(demo, [lips_points], isClosed=False, color=color, thickness=1)
        demo=cv2.GaussianBlur(demo,(5,5),2)
        image=cv2.addWeighted(image,1,demo,0.5,0)
        return image

# Function to convert hex color to RGB format
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        rgb_color = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        return rgb_color

# Function to apply lipstick with improved blending
    def apply_lipstick(image, landmarks, image_width, image_height, color):
        lips_points = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61,
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78
        ]
        mask,lips_points=create_mask(lips_points,landmarks,image_width,image_height,image)
        cv2.fillPoly(mask, [lips_points], color)
        mask=cv2.GaussianBlur(mask,(7,7),4)
        image = cv2.addWeighted(image,1, mask,0.3, 0)
        return image

# Function to apply undereye 
    def apply_undereye(image, landmarks, image_width, image_height, color):
        left_eye_bottom = [133, 155, 154, 153, 145, 144, 163, 7,33]
        right_eye_bottom = [362, 382, 381, 380, 374, 373, 390, 249,263]

        left_eye_bottom_points = np.array(
            [(int(landmarks[point].x * image_width), int(landmarks[point].y * image_height)) for point in left_eye_bottom],
            np.int32)
        right_eye_bottom_points = np.array(
            [(int(landmarks[point].x * image_width), int(landmarks[point].y * image_height)) for point in right_eye_bottom],
            np.int32)
        demo=np.zeros(image.shape, dtype=np.uint8)
        cv2.polylines(demo, [left_eye_bottom_points], isClosed=False, color=color, thickness=1)
        cv2.polylines(demo, [right_eye_bottom_points], isClosed=False, color=color, thickness=1)
        demo=cv2.GaussianBlur(demo,(5,5),2)
        image=cv2.addWeighted(image,1,demo,0.5,0)
        return image

# Function to apply blush with improved blending
    def apply_blush(image, landmarks, image_width, image_height, color):
       blush1_points = [
            280,411,371,352,345
        ]
       mask,blush1_points=create_mask(blush1_points,landmarks,image_width,image_height,image)
       cv2.fillPoly(mask, [blush1_points], color)
       mask = cv2.GaussianBlur(mask, (35, 35), 30)
       image = cv2.addWeighted(image,1, mask, 0.15, 0)
       blush2_points = [
           187,147,137,116,50
           ]
       mask,blush2_points=create_mask(blush2_points,landmarks,image_width,image_height,image)
       cv2.fillPoly(mask, [blush2_points], color)
       mask = cv2.GaussianBlur(mask, (35, 35), 30)
       image = cv2.addWeighted(image,1, mask, 0.15, 0)
       return image
    
# Function to apply concealer  
    def apply_concealer(image, landmarks, image_width, image_height, color,concealer_n):
        concealer_points = [
            83,18,313,421,428,396,175,171,208,201
        ]
        if concealer_n==2:
            concealer_points = [
                1,45,51,3,196,122,193,108,151,337,417,351,419,248,281,275
            ]
        if concealer_n==3:
            concealer_points = [
                412,277,266,280,345,454,356,249,390,373,374,380,381,382,362
            ]
            mask,concealer_points=create_mask(concealer_points,landmarks,image_width,image_height,image)
            cv2.fillPoly(mask, [concealer_points], color)
            mask = cv2.GaussianBlur(mask, (35, 35), 35)
            image = cv2.addWeighted(image, 1, mask, 0.15, 0)
            concealer_points=[133, 155, 154, 153, 145, 144, 163, 7,33,34,227,116,50,36,47,188]
        mask,concealer_points=create_mask(concealer_points,landmarks,image_width,image_height,image)
        cv2.fillPoly(mask, [concealer_points], color)
        mask = cv2.GaussianBlur(mask, (35, 35), 40)
        image = cv2.addWeighted(image, 1, mask, 0.15, 0)
        return image

#function to apply eyeshadow
    def apply_eyeshadow(image, landmarks, image_width, image_height,color):
        eyeshadow_points = [
            414,286,258,257,259,467,445,444,443,442,441
            ]
        mask,eyeshadow_points=create_mask(eyeshadow_points,landmarks,image_width,image_height,image)
        cv2.fillPoly(mask, [eyeshadow_points], color)
        mask = cv2.GaussianBlur(mask, (25, 25), 15)
        image = cv2.addWeighted(image,1, mask, 0.6, 0)
        eyeshadow_points = [
            190,56,28,27,29,30,225,224,223,222,221
            ]
        mask,eyeshadow_points=create_mask(eyeshadow_points,landmarks,image_width,image_height,image)
        cv2.fillPoly(mask, [eyeshadow_points], color)
        mask = cv2.GaussianBlur(mask, (25, 25), 15)
        image = cv2.addWeighted(image,1, mask, 0.6, 0)
        return image


# Function to apply eyebrows with improved blending
    def apply_eyebrows(image, landmarks, image_width, image_height, color):
        eyebrow_points = [
            55,107,66,105,63,70,53,52,55
        ]
        mask,eyebrow_points=create_mask(eyebrow_points,landmarks,image_width,image_height,image)
        cv2.fillPoly(mask, [eyebrow_points], color)
        mask = cv2.GaussianBlur(mask, (25, 25), 15)
        image = cv2.addWeighted(image, 1, mask, 0.2, 0)
        eyebrow_points = [
            285,295,282,283,276,293,334,296,336
        ]
        mask,eyebrow_points=create_mask(eyebrow_points,landmarks,image_width,image_height,image)
        cv2.fillPoly(mask, [eyebrow_points], color)
        mask = cv2.GaussianBlur(mask, (25,25), 15)
        image = cv2.addWeighted(image, 1, mask, 0.2, 0)
        return image

# Function to apply skin toner with improved blending
    def apply_skin_toner(image, landmarks, image_width, image_height, color): 
        skin_points = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        mask,skin_points=create_mask(skin_points,landmarks,image_width,image_height,image)
        cv2.fillPoly(mask, [skin_points], color)
        mask = cv2.GaussianBlur(mask, (11,11), 10)
        return cv2.addWeighted(image, 1, mask,0.1, 0)

#########################################################################################################################################
    st.sidebar.title("Select Makeup Products and Colors")

# Main function
    def main():
        #if lipstick:
        lipstick = st.sidebar.checkbox("Lipstick")
        lipstick_color = st.sidebar.color_picker("Lipstick Color", "#FF0000")
        lipstick_color = hex_to_rgb(lipstick_color)
    
        concealer=st.sidebar.checkbox("Concealer")
        concealer_color=st.sidebar.color_picker("Concealer Color", "#834E0D")
        concealer_color = hex_to_rgb(concealer_color)
        if concealer:
            status=st.radio("Select Place to apply Concealer:",('Chin','Nose','Under Eye'))
            if status=='Chin':
                concealer_n=1
            if status=='Nose':
                concealer_n=2
            if status=='Under Eye':
                concealer_n=3

        undereye = st.sidebar.checkbox("Undereye")
        #if undereye:
        eye_liner_color = st.sidebar.color_picker("Undereye", "#000000")
        eye_liner_color = hex_to_rgb(eye_liner_color)

        blush = st.sidebar.checkbox("Blush")
        #if blush:
        blush_color = st.sidebar.color_picker("Blush Color", "#AA0D70")
        blush_color = hex_to_rgb(blush_color)

        lipliner = st.sidebar.checkbox("Lipliner")
        #if lipliner:
        lipliner_color = st.sidebar.color_picker("Lipliner Color", "#874511")
        lipliner_color = hex_to_rgb(lipliner_color)

        eyebrows = st.sidebar.checkbox("Eyebrows")
        #if eyebrows:
        eyebrow_color = st.sidebar.color_picker("Eyebrow Color", "#000000")
        eyebrow_color = hex_to_rgb(eyebrow_color)

        toner = st.sidebar.checkbox("Skin Toner")
        #if toner:
        toner_color = st.sidebar.color_picker("Skin Toner Color", "#A4620A")
        toner_color = hex_to_rgb(toner_color)
        
        #if eyeshadow:
        eyeshadow= st.sidebar.checkbox("Eye Shadow")
        eyeshadow_color= st.sidebar.color_picker("Eye Shadow Color","#A00E10")
        eyeshadow_color= hex_to_rgb(eyeshadow_color)

        run = st.toggle('Run')
        ba=st.toggle('Before/After')
        FRAME_WINDOW = st.image([])
        frame_file=st.file_uploader('Upload an image...',type=['jpg','jpeg','png'])
        if frame_file:
            if run:
                frame =Image.open(frame_file)
                frame=frame.resize((600,500))
                frame=np.array(frame)
                orig=frame
                image_height, image_width, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        landmarks = face_landmarks.landmark

                        if toner:
                            frame = apply_skin_toner(frame, landmarks, image_width, image_height, toner_color)
                        if concealer:
                            frame=apply_concealer(frame, landmarks, image_width, image_height, concealer_color,concealer_n)
                        if undereye:
                            frame = apply_undereye(frame, landmarks, image_width, image_height, eye_liner_color)
                        if blush:
                            frame = apply_blush(frame, landmarks, image_width, image_height, blush_color)
                        if lipliner:
                            frame = apply_lipliner(frame, landmarks, image_width, image_height, lipliner_color)
                        if eyebrows:
                            frame = apply_eyebrows(frame, landmarks, image_width, image_height, eyebrow_color)
                        if eyeshadow:
                            frame= apply_eyeshadow(frame,landmarks,image_width,image_height,eyeshadow_color)
                        if lipstick:
                            frame = apply_lipstick(frame, landmarks, image_width, image_height, lipstick_color)

                if ba:
                    orig=cv2.resize(orig,(800,500))
                    frame=cv2.resize(frame,(800,500))
                    image_=np.zeros(orig.shape,np.uint8)
                    height,width,_=orig.shape
                    w=width//2
                    h=height
                    smaller_frame=cv2.resize(frame,(0,0),fx=0.5,fy=1)
                    smaller_orig=cv2.resize(orig,(0,0),fx=0.5,fy=1)
                    image_[:h,:w]=smaller_orig
                    image_[:h,w:]=smaller_frame
                    FRAME_WINDOW.image(image_)
                else:
                    FRAME_WINDOW.image(frame)


    main()

if status1=='MAKEUP TRANSFER':
    
    detector = dlib.get_frontal_face_detector() 
    sp = dlib.shape_predictor("C:/Users/Ravindra.Jain/shape_predictor_68_face_landmarks.dat") # path of shape_predictor_68_face_landmarks.dat
    img = dlib.load_rgb_image("01.jpg") # path of 01.jpg
    plt.figure(figsize = (16, 10)) 
    #plt.imshow(img)
    img_result = img.copy()
    dets = detector(img, 1)

    if len(dets) == 0:
        print('cannot find faces!')
    
    fig, ax = plt.subplots(1, figsize=(16, 10))
    for det in dets:
        x, y, w, h = det.left(), det.top(), det.width(), det.height()
    
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    fig, ax = plt.subplots(1, figsize=(16, 10))
    objs = dlib.full_object_detections()

    for detection in dets:
        s = sp(img, detection) # 얼굴의 랜드마크를 찾는다
        objs.append(s)
    
        for point in s.parts():
            circle = patches.Circle((point.x, point.y), radius=3, edgecolor='r', facecolor='r') # patches.Circle : 원을 그린다
            ax.add_patch(circle)

    faces = dlib.get_face_chips(img, objs, size=256, padding=0.3)
    fig, axes = plt.subplots(1, len(faces)+1, figsize=(20, 16))

#face extraction and alignment
    def align_faces(img):
        dets = detector(img, 1)
        objs = dlib.full_object_detections()
    
        for detection in dets:
            s = sp(img, detection)
            objs.append(s)
        
        faces = dlib.get_face_chips(img, objs, size=256, padding=0.35)
        return faces

# test
    test_img = dlib.load_rgb_image("02.jpg") path of # 02.jpg
    test_faces = align_faces(test_img)
    fig, axes = plt.subplots(1, len(test_faces)+1, figsize=(20, 16))

    tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
#loading pre trained model
    saver = tf.compat.v1.train.import_meta_graph("model/checkpoint/model.meta") # path of model.meta
    saver.restore(sess, tf.train.latest_checkpoint("model/checkpoint")) # path of checkpoint folder
    graph = tf.compat.v1.get_default_graph()
    X = graph.get_tensor_by_name('X:0') # source
    Y = graph.get_tensor_by_name('Y:0') # reference
    Xs = graph.get_tensor_by_name('generator/xs:0') # output
#image normalization
    def preprocess(img):
        return img.astype(np.float32) / 127.5-1.
    def postprocess(img):
        return ((img + 1.) * 127.5).astype(np.uint8)
    def color(color_options):
        selected_color_name = st.sidebar.selectbox("Select a color:", options=color_options)
        return selected_color_name
######################################################################################################
#def main():
    run=st.toggle("RUN FOR MAKEUP")
    upload=st.sidebar.file_uploader("UPLOAD MAKEUP", type="jpg")
    if upload is not None:
        makeup=Image.open(upload)
        st.image(makeup,width=250, caption='Uploaded Makeup')

    while run:
        uploaded_file = st.sidebar.file_uploader("Upload an image...", type="jpg")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image,width=250,caption='Uploaded Image')
            img1=np.array(image)
            img1_faces = align_faces(img1)

            img2 = np.array(makeup)
            img2_faces = align_faces(img2)

            fig, axes = plt.subplots(1, 2, figsize=(16, 10))

            src_img = img1_faces[0]
            ref_img = img2_faces[0]

            X_img = preprocess(src_img)
            X_img = np.expand_dims(X_img, axis=0)

            Y_img = preprocess(ref_img)
            Y_img = np.expand_dims(Y_img, axis=0)

            output = sess.run(Xs, feed_dict={
                X: X_img,
                Y: Y_img
            })
            output_img = postprocess(output[0])
            fig, axes = plt.subplots(1, 3, figsize=(20, 10))
            axes[0].set_title('Source')
            st.image(output_img,width=500,caption='Image with Makeup')
        break
