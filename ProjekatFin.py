# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:45:23 2019

@author: Nikola Mandic
"""

import numpy as np
import cv2
import os
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.svm import SVC # SVM klasifikator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import mytracker as track




def detect_line(img,zelena=True):
    # detekcija koordinata linije koristeci Hough transformaciju
    
    if zelena: 
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges_img = cv2.Canny(gray_img, 180, 255, apertureSize=3)
        # minimalna duzina linije
        min_line_length = 200
        # Hough transformacija
        lines = cv2.HoughLinesP(image=edges_img, rho=1, theta=np.pi/180, threshold=10, lines=np.array([]),
                                minLineLength=min_line_length, maxLineGap=20)
        
        a,b,c = lines.shape
        
        x1 = lines[0][0][0]
        y1 = 480 - lines[0][0][1]
        x2 = lines[0][0][2]
        y2 = 480 - lines[0][0][3]
    else:
        
        blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
        hsv = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)
        low_blue = np.array([110, 50, 50])
        up_blue = np.array([130, 255, 255])
        
        mask = cv2.inRange(hsv, low_blue, up_blue)
        
        edges_img = cv2.Canny(mask,75,150)
        # minimalna duzina linije
        min_line_length = 200
        
        # Hough transformacija
        lines = cv2.HoughLinesP(image=edges_img, rho=1, theta=np.pi/180, threshold=10, lines=np.array([]),
                                minLineLength=min_line_length, maxLineGap=20)
        
        a,b,c = lines.shape
        
        x1 = lines[0][0][0]
        y1 = 480 - lines[0][0][1]
        x2 = lines[0][0][2]
        y2 = 480 - lines[0][0][3]
    
    return (x1, y1, x2, y2)    

def process_video(video_path):
    sum_of_nums = 0
    tracker = None
    flag = False
    
    # ucitavanje videa
    frame_num = 0
    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_num) # indeksiranje frejmova
# analiza videa frejm po frejm
    while True:
        frame_num += 1
        ret_val, frame = cap.read()
        # plt.imshow(frame)
        # ako frejm nije zahvacen
        if not ret_val:
            break
    
        #detekcija ukoliko su brojevi presli preko linije 
        if frame_num == 1: # ako je prvi frejm, detektuj liniju
            line_coords_blue = detect_line(frame,False)
            line_coords_green = detect_line(frame, True)
            tracker = track.Tracker(line_coords_blue,line_coords_green)
            
            print ('blue: '+str(line_coords_blue))
            print ('green: ' + str(line_coords_green))
        
        if frame_num >= 2:
            flag = True
        
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #plt.imshow(frame_gray, "gray")
        ret, frame_bin = cv2.threshold(frame_gray, 170, 255, cv2.THRESH_BINARY)
        #plt.imshow(frame_bin, "gray")
        frame_numbers = cv2.dilate(frame_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=2)
        #plt.imshow(frame_numbers, "gray")
        contours, aa = cv2.findContours(frame_numbers.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
        number_contours = []
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            if w > 11 or h > 11:
                number_contours.append(cv2.boundingRect(contour))
        
        if frame_num == 1:
            tracker.start(number_contours, frame_numbers)
        else :
            tracker.update(number_contours,frame_numbers, flag)
            
    
    numbers = tracker.numbers
    cntAdd = 0
    cntDec = 0
    
    for number in numbers.values():
        if(not number.forbidden):
            if(number.add):
                cntAdd += 1
                sum_of_nums += number.value
            if(number.decrease):
                cntDec += 1
                sum_of_nums -= number.value
    
    cap.release() 
    return sum_of_nums


path = os.path.abspath('.')
video_path = path + "\\\\video-9.avi"
print(video_path)

print(process_video(video_path))


