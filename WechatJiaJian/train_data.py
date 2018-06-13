#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 12:41:26 2018

@author: llq
"""
'''
from main import contour_split,narrow_picture,extend_pixs
import numpy as np


#Configure
pixs=28
loc={"top_x":186,
     "top_y":398,
     "bottom_x":474,
     "bottom_y":618,
     "mouse_corrcet_x":209,
     "mouse_corrcet_y":819,
     "mouse_wrong_x":456,
     "mouse_wrong_y":819,}
         
path1="./screenshot/temp1.png"

#read image
image_left,image_right,img=contour_split(path1,loc)
del image_left[1]

#narrow picture
narrow_picture(image_left,pixs)
narrow_picture(image_right,pixs)

#extend pixs to 28*28
extend_pixs(image_left,pixs)
extend_pixs(image_right,pixs)
image_left=np.float32(np.array(image_left))
image_right=np.float32(np.array(image_right))

""""""
path2="./screenshot/temp2.png"

image_left2,image_right2,img=contour_split(path2,loc)
del image_left2[1]

#narrow picture
narrow_picture(image_left2,pixs)
narrow_picture(image_right2,pixs)

#extend pixs to 28*28
extend_pixs(image_left2,pixs)
extend_pixs(image_right2,pixs)
image_left2=np.float32(np.array(image_left2))
image_right2=np.float32(np.array(image_right2))

""""""
path3="./screenshot/temp3.png"

image_left3,image_right3,img=contour_split(path3,loc)
del image_left3[1]

#narrow picture
narrow_picture(image_left3,pixs)
narrow_picture(image_right3,pixs)

#extend pixs to 28*28
extend_pixs(image_left3,pixs)
extend_pixs(image_right3,pixs)
image_left3=np.float32(np.array(image_left3))
image_right3=np.float32(np.array(image_right3))


""""""
path4="./screenshot/temp4.png"

image_left4,image_right4,img=contour_split(path4,loc)
del image_left4[1]

#narrow picture
narrow_picture(image_left4,pixs)
narrow_picture(image_right4,pixs)

#extend pixs to 28*28
extend_pixs(image_left4,pixs)
extend_pixs(image_right4,pixs)
image_left4=np.float32(np.array(image_left4))
image_right4=np.float32(np.array(image_right4))

""""""
path5="./screenshot/temp5.png"

image_left5,image_right5,img=contour_split(path5,loc)
del image_left5[1]

#narrow picture
narrow_picture(image_left5,pixs)
narrow_picture(image_right5,pixs)

#extend pixs to 28*28
extend_pixs(image_left5,pixs)
extend_pixs(image_right5,pixs)
image_left5=np.float32(np.array(image_left5))
image_right5=np.float32(np.array(image_right5))

""""""
path6="./screenshot/temp6.png"

image_left6,image_right6,img=contour_split(path6,loc)
del image_left6[1]

#narrow picture
narrow_picture(image_left6,pixs)
narrow_picture(image_right6,pixs)

#extend pixs to 28*28
extend_pixs(image_left6,pixs)
extend_pixs(image_right6,pixs)
image_left6=np.float32(np.array(image_left6))
image_right6=np.float32(np.array(image_right6))

""""""
image_left=np.concatenate((image_left,image_left2,image_left3,image_left4,image_left5,image_left6),axis=0)
image_right=np.concatenate((image_right,image_right2,image_right3,image_right4,image_right5,image_right6),axis=0)

image_left=np.concatenate((image_left,image_right),axis=0)
image_left=np.reshape(image_left,(-1,1,28,28))
label=np.array([8,7,4,3,7,6,3,7,6,6,3,4,1,5,1,3,1,0,1,2,9])
np.save("./screenshot/train_data.npy",image_left)
np.save("./screenshot/label.npy",label)
'''
import os
loc = {
    "top_x":186,
    "top_y":398,
    "bottom_x":474,
    "bottom_y":618
}
command = 'import -window root -crop {0}x{1}+{2}+{3} screenshot.png'
command = command.format(loc['bottom_x'] - loc['top_x'],
                         loc['bottom_y'] - loc['top_y'],
                         loc['top_x'],
                         loc['top_y'])
os.system(command)