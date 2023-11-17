import numpy as np
from scipy.spatial import distance
import cv2
import dlib

class data_distance:
    #calculates the EAR using the Euclidean distance 
    #for eyes 
    def eye_aspect_ratio(self, eye):
        print(eye)
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A+B) / (2*C)
        return ear


    #calculates the MAR
    #for mouse
    def mouth_aspect_ratio(self, mouth):
        A = np.linalg.norm(mouth[14] - mouth[18])
        B = np.linalg.norm(mouth[12] - mouth[16])
        mar = A / B
        return mar


    def get_head_tilt_angle(self, head):
        left_eye = head[36:42]
        right_eye = head[42:48]

        dY = right_eye[:, 1].mean() - left_eye[:, 1].mean()
        dX = right_eye[:, 0].mean() - left_eye[:, 0].mean()

        angle = np.degrees(np.arctan2(dY, dX))
        return angle
    

"""
    def lip_aspect_ratio(self, shape):
        mouth = shape[self.mStart:self.mEnd]
        A = distance.euclidean(shape[62], shape[66])
        B = distance.euclidean(shape[63], shape[65])
        C = distance.euclidean(mouth[0], mouth[6])
        lip = (A+B) / (2*C)
        return lip

    def head_tilt(self, shape):
        left_eye = shape[self.lStart:self.lEnd][0]
        right_eye = shape[self.rStart:self.rEnd][0]
        mouth = shape[self.mStart:self.mEnd][0]

        angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]) - np.arctan2(mouth[1] - left_eye[1], mouth[0] - left_eye[0])
        angle = np.degrees(angle)
        return angle
"""
class HeadTiltAnalyzer:
    def __init__(self):
        self.lStart = 36  # Left eye landmark start index
        self.lEnd = 42    # Left eye landmark end index
        self.rStart = 42  # Right eye landmark start index
        self.rEnd = 48    # Right eye landmark end index

    def head_tilt_angle(self, shape):
        left_eye = shape[self.lStart:self.lEnd][0]
        right_eye = shape[self.rStart:self.rEnd][0]
        nose = shape[27]

        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        dY = nose[1] - (left_eye[1] + right_eye[1]) // 2
        dX = nose[0] - (left_eye[0] + right_eye[0]) // 2
        tilt = np.degrees(np.arctan2(dY, dX))
        return angle, tilt


class scale:
    def rescale_frame(self, frame, percent):
        width = int(frame.shape[1] * percent/ 100)
        height = int(frame.shape[0] * percent/ 100)
        dim = (width, height)
        resize_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        return resize_frame

