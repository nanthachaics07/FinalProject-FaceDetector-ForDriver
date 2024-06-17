from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import time
import dlib
import pygame
import cv2
import csv
import datetime
import os  # Add this line to import the 'os' module

# Function to ensure the folder exists
def ensure_folder_exists(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

#Def_data.py (code DEF without this file)
from Def_data import data_distance , scale

dataDT = data_distance()
sc = scale()

# 20 : 1 [1 = Sec]

EYE_ART = 0.3   #eye_dis
EYE_ARCF = 50   #eye_time s/f

EYE_OPEN_CONSEC_FRAMES = 40 #bewilder_time s/f

BLINK_THRESHOLD = 0.2   #eye_blink_dis 
BLINK_CONSEC_FRAMES = 1 #eye_blink_timr s/f

FACE_ARCF = 35  #face_degree s/f

MOUTH_AR = 0.3  #mouth/yawn_dis
YAWN_ARCF = 15  #mouth/yawn_time s/f
"""
#old head tilt angle !!!!!!!!!!!!!!!!!!!!!
HEAD_TILT_THRESHOLD = 20    #tilted neck_dis
HEAD_TILT_CONSEC_FRAMES = 80    #tilted neck_time s/f
"""
#New head tilt angle
TILT_ANGLE_THRESHOLD = 15
DETECTION_DURATION = 4  #1 = 1 sec
#Initialize variables for head tilt detection
start_time = None
tilt_detected = False


#"The Count" for alert if variable +=1
#แยกเพื่อความสบายใจของเซฟ
#!!!!! Do not change !!!!!!
EYE_COUNTER = 0
EYE_OPEN_COUNTER = 0
BLINK_COUNT = 0
YAWN_COUNTER = 0
COUNTER_FACE = 0
HEAD_COUNTER = 0
TOTAL = 0

#Sound alert
pygame.mixer.init()
pygame.init()
pygame.mixer.music.load('audio/iSUS.mp3')  #if used another sound (!!change namein " ")

'''
#Mr.SAFE! 'Copy this fix from stackoverflow' for change sound lag and another sound problem
pygame.mixer.pre_init(22050, -16, 2, 1024)
pygame.mixer.init(22050, -16, 2, 1024)
pygame.mixer.init(2048)
'''

#Training file 68 distance
# face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

face_cascade = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_default.xml")
detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape68coordinates/shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor("./shape68coordinates/shape_predictor_68_face_landmarks.dat")


#Imutils library
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


#Setting cam devices
#   "0" meaning realtime cam
# Used ", cv2.CAP_FFMPEG" when vdo.mp4 lag // which can provide hardware-accelerated video processing and potentially improve performance
video_capture = cv2.VideoCapture(0)
#video_capture = cv2.VideoCapture('video\day1.mp4', cv2.CAP_FFMPEG)

"""
#Check if video capture is successfully opened
if not video_capture.isOpened():
    print("Failed to open video file.")
    exit()
"""

#Early time for when start cam 
time.sleep(1.0)


#Count for break the while loop if cam mistake
count=0

# Initialize variables for FPS calculation
start_time = time.time()
frame_count = 0

folder_name = 'ALERT_DATA'  # The name of the folder to store CSV files
ensure_folder_exists(folder_name)  # Ensure the folder exists

csv_file = os.path.join(folder_name, "alerts.csv")  # Use os.path.join to create the full file path


#Big while loops for all condition
while(True):
    #separate cam read to 'two' variable
    ret, frame = video_capture.read()

    # Increment the frame count
    frame_count += 1
    # Calculate FPS
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    # Display the FPS on the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (300, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    # Reset frame count and start time if more than 2 seconds have passed
    if elapsed_time > 2:
        start_time = time.time()
        frame_count = 0



    # scale of cam output
    frame = sc.rescale_frame(frame, percent=100)

    #Framee mistake
    #เขียนรวมล่างได้แต่อยากเเยกเพื่อความสบายใจของเซฟ โอเคนะ
    if not ret:
        print("Ignoring empty camera frame")
        continue

    #Same upper code, It shell show "str" alert in line 88
    #For frame mistake again
    count=count+1  
    if ret == False:
        break

    """
    #Variable of line 83 
    #resize
    resized_frame = cv2.resize(frame, cv2_size)
    """

    #change color RGB to Gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    #detect face for building face RECT..
    face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    #loop iterates of face rectangles
    for (x,y,w,h) in face_rectangle:
        #Extracts the region of interest from the frame based on the coordinates 
        #ก็แยกพิกัดในเฟรมในลิสอะหนุ่ม จารอินเดียเขาสอนแบบนี้เลย เอามาแปลง
        face = frame[y:y+h, x:x+w]

        #Wide, High //2 for find the center of face_RECT 
        center = (x + w//2, y + h//2)

        #RECT output on cam
        #Draws a color rectangle around the current face rectangle
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        #loops grayface(BGR) of line 108
        for face in faces:

            #facial landmark predictor tto detect coordinates in BGR
            shape = predictor(gray, face)

            #Converts the detected facial landmarks to a Numpy array
            #แปลงเป็น np 
            shape = face_utils.shape_to_np(shape)

            #FOR Calculate the head tilt angle ONLYYYYY!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            angle = dataDT.get_head_tilt_angle(shape)
            
            
            #Extracts the left and right eye landmarks
            left_eye = shape[36:42]
            right_eye = shape[42:48]

            #Calculates the center point between L/R eyes
            #ที่จิงจะใช้ for loop แต่บังบอกว่าใช่บวกเอาแล้วหาร 2
            eyes_center = (left_eye.mean(axis=0) + right_eye.mean(axis=0)) / 2

            #คำนวณมุมของเส้นที่เชื่อมตรงกลางของตาซ้าย/ขวา
            dy = right_eye[1][1] - left_eye[1][1]
            dx = right_eye[1][0] - left_eye[1][0]
            angle = np.degrees(np.arctan2(dy, dx))
            
            #สร้าง2จุดคำนวน
            head_pose = (eyes_center, angle)
            
            #Draws a circle at the eyes center on the frame
            #ไอจุดกลมแดงๆตรงกลางระหว่างตาอะ เปลืองramแต่ก็จะใส่เดี๋ยงเอาออกจารสั่ง
            x, y = head_pose[0]
            #cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.putText(frame, "Head angle: {:.2f} degrees".format(head_pose[1]), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            """
            #iterates over each (x, y) coordinate in the variable "shape"
            for (x, y) in shape:

                #จุดกลมๆ 68 จุด (shape_predictor_68_face_landmarks)
                cv2.circle(frame, (x, y), 1, (255, 0, 255), 2)
            """

            #ใช้ลิสแยกจุดสังเกตุ L/R eyes , mouth 
            #extracts the landmarks corresponding to the left eye, right eye, and mouth from the "shape" variable
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]

            #calculates the eye aspect ratio from 'def eye_aspect_ratio'
            leftEyeAspectRatio = dataDT.eye_aspect_ratio(leftEye)
            rightEyeAspectRatio = dataDT.eye_aspect_ratio(rightEye)

            # /2 twice eye results
            eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2

            #calculates the mouth aspect ratio from 'def mouth_aspect_ratio'
            mouthAspectRatio = dataDT.mouth_aspect_ratio(mouth)

            #computes the convex hulls of the eyes L/R, mouth from cv2
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mouthHull = cv2.convexHull(mouth)

            #output color of upper code
            cv2.drawContours(frame, [leftEyeHull, rightEyeHull, mouthHull], -1, (0, 255, 0), 1)

            if len(face_rectangle) == 0:
                COUNTER_FACE += 1
                if COUNTER_FACE >= FACE_ARCF:
                    pygame.mixer.music.play(0)
                    pygame.time.wait(1000)
                    cv2.putText(frame, "!!!Face Not Detected!!!", (70,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,80,255), 2)
                    
                    timestamp = datetime.datetime.now()
                    alert_message = f"!!!Face Not Detected!!! {timestamp}"

                    # Write the alert message to the CSV file
                    writer.writerow([timestamp, alert_message])
            
            else:
                pygame.mixer.music.stop()
                COUNTER_FACE = 0

            #The condition of single eye missing
            # L or R == 0 = eyemis... and then alert
            if (leftEyeAspectRatio == 0 or rightEyeAspectRatio == 0):
                pygame.mixer.music.play(0)
                pygame.time.wait(1000)
                cv2.putText(frame, "!!!Single Eye MIS!!!", (70,100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,80,255), 2)

                timestamp = datetime.datetime.now()
                alert_message = f"!!!Single Eye MIS!!! {timestamp}"

                # Write the alert message to the CSV file
                writer.writerow([timestamp, alert_message])

            else:
                pygame.mixer.music.stop()
            

            #The condition of both eyes missing
            #If eyes coordinates <= 0.3 it shell alert
            if(eyeAspectRatio < EYE_ART):
                EYE_COUNTER += 1

                #If EC >= 20 it shell alert
                if EYE_COUNTER >= EYE_ARCF:
                    pygame.mixer.music.play(0)
                    pygame.time.wait(1000)
                    cv2.putText(frame, "!!!Eye MIS!!!", (70,100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,80,255), 2)
                    
                    timestamp = datetime.datetime.now()
                    alert_message = f"!!!Eye MIS!!! {timestamp}"

                    # Write the alert message to the CSV file
                    writer.writerow([timestamp, alert_message])

            else:
                EYE_COUNTER = 0
                pygame.mixer.music.stop()

            #The condition of bewilder
            if(eyeAspectRatio < EYE_ART):
                EYE_OPEN_COUNTER = 0

            else:
                EYE_OPEN_COUNTER += 1
                #Here! this code for alert if bewilder
                if EYE_OPEN_COUNTER >= EYE_OPEN_CONSEC_FRAMES:
                    pygame.mixer.music.play(0)
                    pygame.time.wait(1000)
                    cv2.putText(frame, "!!! BEWILDER !!!", (70,100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,80,255), 2)
                    timestamp = datetime.datetime.now()
                    alert_message = f"!!! BEWILDER !!! {timestamp}"

                    # Write the alert message to the CSV file
                    writer.writerow([timestamp, alert_message])
                                

            #The condition of gape
            if(mouthAspectRatio > MOUTH_AR):
                YAWN_COUNTER += 1

                if YAWN_COUNTER >= YAWN_ARCF:
                    pygame.mixer.music.play(0)
                    pygame.time.wait(1000)
                    cv2.putText(frame, "!!! GAPE !!!", (70,100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,80,255), 2)
                    
                    timestamp = datetime.datetime.now()
                    alert_message = f"!!! GAPE !!! {timestamp}"

                    # Write the alert message to the CSV file
                    writer.writerow([timestamp, alert_message])

            else:
                YAWN_COUNTER = 0
                pygame.mixer.music.stop()

            
            # Perform head tilt detection
            if start_time is None:
                start_time = time.time()
            elif time.time() - start_time >= DETECTION_DURATION and not tilt_detected:
                if abs(angle) > TILT_ANGLE_THRESHOLD:
                    pygame.mixer.music.play(0)
                    pygame.time.wait(1000)
                    tilt_detected = True
                    cv2.putText(frame, "!!!Head Tilt Angle!!!", (70,100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,80,255), 2)

                    timestamp = datetime.datetime.now()
                    alert_message = f"!!!Head Tilt Angle!!! {timestamp}"

                    # Write the alert message to the CSV file
                    writer.writerow([timestamp, alert_message])

            else: 
                pygame.mixer.music.stop()

            if(leftEyeAspectRatio < BLINK_THRESHOLD and rightEyeAspectRatio < BLINK_THRESHOLD):
                BLINK_COUNT += 1

            else:
                if BLINK_COUNT >= BLINK_CONSEC_FRAMES :
                    TOTAL += 1
                BLINK_COUNT = 0
                cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(frame, "EAR: {:.2f}".format(eyeAspectRatio), (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "MOUSE: {:.2f}".format(mouthAspectRatio), (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    """
    #displays the resized
    cv2.imshow("resized_frame", resized_frame)
    """

    #If enter "Q" 's mean break the loops and exited state
    cv2.imshow('Video', frame)

    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break


#To clear cam device
video_capture.release()

#closes all OpenCV
cv2.destroyAllWindows()