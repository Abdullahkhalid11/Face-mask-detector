#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[5]:


import os
import cv2
import argparse
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# In[ ]:


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--video", required=True,
                help="path to video")
ap.add_argument("-m", "--model", required=True,
                default = "mask_detector.model",
                help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type = float,default = 0.80,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


## object created of the MTCNN face detector
detector = MTCNN()

## load the mask detection model
model = load_model(args["model"])

## video_path : path to the video file you want to run test on.
video_path = args["video"]

## create a videoCapture object from the video file.
capture9 = cv2.VideoCapture(video_path)


while True:
    
    ## read the frame
    has_frame, frame = capture9.read()
    
    ## if there are no more frames then this means that the video has reached the end.
    if not has_frame:
        print('Reached the end of the video')
        break

    ## converting the frame fetched from teh video file from BGR to RGB 
    ## bcz MTCNN gives better performance on RGB images.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    ## resizing because this video is 1080p and i like to keep my CPU less throttle baqi aap ki mrzi. ;)
    image = cv2.resize(image,(320,256))
    
    ## detecting faces from the frame.
    faces = detector.detect_faces(image)

    ## iterating through every face.
    for face in faces:
        
        ## face['box'] gives the bounding boxes of all the faces.
        bounding_box = face['box']
        
        ## face['keypoints'] gives the keypoints of eyes, nose and mouth ends.
        keypoints = face['keypoints']

        ## setting a threshould for confidence of the face detected.
        if face['confidence'] > args["confidence"]:
            
            ## marking the keypoints.
            cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
            cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
            cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
            cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
            cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)

        ## padding the face boundry. because of past bad experiences and a little bit of gut feeling.
        padding = 10
        startX, startY, w, h = [0 if i < 0 else i for i in bounding_box]
        startX, startY, w, h = startX -padding , startY-padding, w+2*padding, h+2*padding
        startX, startY, w, h = [0 if i < 0 else i for i in (startX, startY, w, h)]
        endX, endY = startX + w, startY + h
        
        ## drawing the rectangle around the face.
        cv2.rectangle(image,(startX,startY),(endX,endY),(255,0,0),2)

        ## crop the face from the frame
        face = image[startY:endY, startX:endX]

        ## preprocessing steps required for mask detecting model
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        ## predicitons of the mask detection model
        (mask, withoutMask) = model.predict(face)[0]

        ## creating a label.
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0)

        ## also writing the probabilites of results 
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        cv2.putText(image, label, (startX, startY),cv2.FONT_HERSHEY_SIMPLEX, .7, color, )

    ## writing the number of faces captured and dim on the frame. Mrzi hai na kro tb b sahi,
    dim = "Resolution: {}x{} Faces: {}".format(image.shape[0],image.shape[1], len(faces))
    cv2.putText(image,dim,(0,25),cv2.FONT_HERSHEY_SIMPLEX ,1, (0,155,255),2)

    ## displaying the frame. press key q to stop displaying.
    cv2.imshow('img',cv2.cvtColor(image,cv2.COLOR_RGB2BGR))

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
            break
            
## releasing the capturing object.
capture9.release()
## closing the windows.
cv2.destroyAllWindows()

