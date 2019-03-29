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
        self.decrease = False
        self.overLapping = False
        self.disappeared = False
        self.value = -1
        self.k = 0
        self.n = 0

    def calc_center(self):
        x,y,w,h = self.rect
        cX = round((x+x+w)/ 2.0, 2)
        cY = round((y+y+h)/ 2.0, 2)
        center = (cX,cY)
        return center
    
    #metoda za racunanje potencijalne putanje cifre
    #racunamo kao pravolinijsku putanju koju update svaki frame
    #kriterijum za odredjivanje sledece cifre u pracenju
    def calc_path(self, newRect):
        x1,y1,w,h = newRect
        x2,y2,w,h = self.rect
        
        y1 = 480.0 - y1
        y2 = 480.0 - y2
        
        k = round((y2 - y1)/(x2 - x1),3)
        n = round(y1 - k*x1, 3)
        return k,n

    def update_number_info(self,newRect,k,n):
        self.rect = newRect
        self.k = k
        self.n = n
        return
        
    def update_value(self,newValue):
        if(self.value == -1):
            self.value = newValue
        return

    def add_num(self):
        self.add_num = True
        return
    
    def dec_num(self):
        self.decrease = True
        return

num = Number((1,1,1,1),1)



       