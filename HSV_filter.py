import sys
import cv2
import numpy as np
import time

def add_HSV_filter(frame, camera):

    #blurring the frame
    blur = cv2.GaussianBlur(frame,(5,5),0)

    #Converting RGB to HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    l_b_r = np.array([60,110,50]) # lower limit for red ball
    u_b_r = np.array([255,255,255]) # upper limit for red ball
    l_b_l = np.array([143, 110, 50]) # lower limit for red ball
    u_b_l = np.array([255,255,255]) # upper limit for red ball

    if(camera == 1):
        mask = cv2.inRange(hsv, l_b_r, u_b_r)
    else:
        mask = cv2.inRange(hsv, l_b_l, u_b_l)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    return mask        

