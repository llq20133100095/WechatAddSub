#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 12:49:45 2018

@author: llq
"""
import cv2
import os

#Screenshot
os.system("import -window root ./screenshot/temp.png")

#
sc_img=cv2.imread("./screenshot/temp.png")
loc={"top_x":186,"top_y":398,"bottom_x":474,"bottom_y":618}
sc_img=sc_img[loc["top_y"]:loc["bottom_y"],loc["top_x"]:loc["bottom_x"]]
cv2.imwrite("./screenshot/1.png",sc_img)