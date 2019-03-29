# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:50:02 2019

@author: Nikola Mandic

Negde se racunaju koordinate kao 480 - y
Dok u nekim slucajevima jednostavno nije pottrebno tako
"""

class Number():
    def __init__(self, rect, ID):
        self.ID = ID
        self.rect = rect
        self.add = False
        self.overLapping = False
        self.value = -1
        self.k = 0
        self.n = 0

    def calc_center(self):
        x,y,w,h = self.rect
        cX = round((x+x+w)/ 2.0, 2)
        cY = round((y+y+h)/ 2.0, 2)
        center = (cX,cY)
        return center
    
    def calc_path(self):
        return

    def check_overLapping(self):
        return




       