#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 18:49:23 2018

@author: llq
"""
import cv2

#read picture
lena=cv2.imread("lena_color.jpg")
lena0=cv2.imread("lena_color.jpg",cv2.IMREAD_GRAYSCALE)

#show picture
cv2.imshow("lena",lena0)
cv2.waitKey(0)
cv2.destroyAllWindows()

#write picture
cv2.imwrite("1.jpg",lena)
