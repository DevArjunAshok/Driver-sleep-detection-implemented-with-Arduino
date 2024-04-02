#libray used for working with images
import cv2
from engi1020.arduino.api import *
from time import sleep
#library used for powerful mathematic

#facial detector library(68 pointts)
#used for basic image processsing(rotating)
from imutils import face_utils# Imports utilities for facial landmark processing.
import numpy as np
import dlib


#initializing camera and taking video instances
cap=cv2.VideoCapture(0)
#cv2.VideoCapture(0): Means first camera or webcam.
#cv2.VideoCapture(1):  Means second camera or webcam.
#cv2.VideoCapture(“file name.mp4”): Means video file
detect=dlib.get_frontal_face_detector()#Initializes Dlib's frontal face detector.
predict=dlib.shape_predictor("shape_predictor_68_face_landmarks .dat")  #Loads the pre-trained model for detecting 68 facial landmarks.
    
sleep=0
drowsiness=0
active=0
status=""
color=(0,0,0)

def distances(a,b):
    dist=np.linalg.norm(a-b)   #to find the euclidean distance between two points
    return dist

#function to check whether eye closed or opened
def blinked(a,b,c,d,e,f):
    short_distance=distances(b,d)+distances(c,e)
    long_distance=distances(a,f)
    ratio=(short_distance)/(2*long_distance)
    if ratio>0.25 :   #0.25 is approximately the ratio for open eye
        return 2
    elif 0.21<ratio<=0.25:  #drowsiness
        return  1
    else:
        return 0      #eye is closed
servo_set_angle(3,0)  
#BUTTON_PRESS_THRESHOLD = False 
while True:

    _,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)   #converts frame to gray color to reduce complexity(gray scaling)
    faces=detect(gray)
    face_frame=frame.copy()

   #if digital_read(6)!=BUTTON_PRESS_THRESHOLD:
        #buzzer_stop(5)
    
    for face in faces: #This loop iterates over a collection of detected faces.
        x1=face.left() #These lines extract the coordinates of the bounding box of the detected face.
        x2=face.right()
        y1=face.top()
        y2=face.bottom()
        
        cv2.rectangle(face_frame,(x1,y1),(x2,y2),(0,255,0),2)
        landmarks=predict(gray,face)
        landmarks=face_utils.shape_to_np(landmarks)     
        
        #gets coordinates for eyes and cclaculates the euclidean distance
        
        left_blink=blinked(landmarks[36],landmarks[37],landmarks[38],landmarks[41],landmarks[40],landmarks[39])
        right_blink=blinked(landmarks[42],landmarks[43],landmarks[44],landmarks[47],landmarks[46],landmarks[45])

        

        #checks whether sleeping or not
        if left_blink==0 or right_blink==0:
            sleep+=1
            drowsiness=0
            active=0
            if sleep>6:
                status="Alert, Sleeping!"
                color=(0,255,0)
                rgb_lcd_clear()
                rgb_lcd_print(f"{status}")
                buzzer_note(5,500,2)
                front_dist=ultra_get_centimeters(6)
                if front_dist<30:
                    servo_set_angle(3,90)
                
        
        elif(left_blink==1 or right_blink==1):
            sleep=0
            drowsiness+=1
            active=0
            if drowsiness>6:
                status="Alert, Drowsy"
                color=(0,0,255)
                rgb_lcd_clear()
                rgb_lcd_print(f"{status}")
        else:
            sleep=0
            drowsiness=0
            active+=1
            if active>6:
                status="Active"
                color=(255,0,0)
                rgb_lcd_clear()
                rgb_lcd_print(f"{status}")
        cv2.putText(frame,status,(100,100),cv2.FONT_HERSHEY_PLAIN,1.2,color,3)
        for n in range(0,68):
            (x,y)=landmarks[n]
            cv2.circle(face_frame,(x,y),1,(255,255,255),-1)
            
    
        
        
        

    cv2.imshow('frame',frame)
    cv2.imshow("Result of detector",face_frame)
    k=cv2.waitKey(25)
    if k==27 or digital_read(6):        #kill switch
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

    