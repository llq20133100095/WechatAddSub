#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 20:09:03 2018

@author: llq
"""

import cv2
import theano.tensor as T
import theano
import lasagne
import numpy as np
from mnist import build_cnn
from pymouse import PyMouse
import time
import os

def insert_sort(data,axis):
    """
    According to the axis, sort the image
    """
    for i in range(1,len(data)):
        for j in reversed(range(1,i+1)):
            if(data[j][axis]<data[j-1][axis]):
                t=data[j]
                data[j]=data[j-1]
                data[j-1]=t

def contour_split(path,loc=False):
    """
    Split the contour in image
    """
    #binary
    img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    if loc!=False:
        #Real picture
        img=img[loc["top_y"]:loc["bottom_y"],loc["top_x"]:loc["bottom_x"]]
        
    ret, img_bin = cv2.threshold(img, 180, 255,cv2.THRESH_BINARY)
    
    #Contour detection: only check the external
    image, contours, hierarchy = cv2.findContours(img_bin,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  
    #cv2.drawContours(img,contours,-1,(0,0,255),3)
    
    #save contour
    data=[]
    for i in range(0,len(contours)):  
        x, y, w, h = cv2.boundingRect(contours[i])   
#        image=cv2.rectangle(image, (x,y), (x+w,y+h), (153,153,0), 5) 
#        cv2.imshow("1",image)
#        cv2.waitKey(0)
    
        #save: x and y
        data.append([x,y,w,h])
    
    cv2.destroyAllWindows()

    #sort the "y" axis       
    insert_sort(data,1)
    
    #get equation left and right
    data=data[:-2]
#    if len(data)==
    data_left=data[:3]
    data_right=data[3:]
    
    #have the order in "x" axis
    insert_sort(data_left,0)
    insert_sort(data_right,0)
    
    #split contour in image
    image_left=[]
    image_right=[]
    for x,y,w,h in data_left:
        #sort the "x" axis
        image_left.append(image[y:y+h,x:x+w]) # 先用y确定高，再用x确定宽
    
    for x,y,w,h in data_right:
        #sort the "x" axis
        image_right.append(image[y:y+h,x:x+w]) # 先用y确定高，再用x确定宽
    
    return image_left,image_right,data
    
def extend_pixs(image,pixs):
    """
    Extend pixs in image.
    Get image in pixs * pixs
    """
    #extend pixs to 28*28
    for i in range(len(image)):
        a_image=image[i]
        #check x axis
        if len(a_image)<pixs:
            a_image=cv2.copyMakeBorder(a_image,(pixs-len(a_image))/2,(pixs-len(a_image))/2+1,0,0,cv2.BORDER_CONSTANT,value=[0])
        #check y axis
        if len(a_image[0])<pixs:
            a_image=cv2.copyMakeBorder(a_image,0,0,(pixs-len(a_image[0]))/2,(pixs-len(a_image[0]))/2+1,cv2.BORDER_CONSTANT,value=[0])
        
        a_image=a_image[:28,:28]
        image[i]=a_image

def sum_and_sub(sum_path,sub_path,loc,pixs):
    """
    Basis picture:SUM   SUB
    Comparise these two picture, in order to predicte the "+" or "-"
    """
    sum_img,_,_=contour_split(sum_path)
    sub_img,_,_=contour_split(sub_path,loc)
    
    narrow_picture(sub_img,pixs)
    
    extend_pixs(sum_img,pixs)
    extend_pixs(sub_img,pixs)
    
    sum_img=sum_img[1]
    sub_img=sub_img[1]

    return sum_img,sub_img
    
def CNN_pre(image_left,image_right,pre_fn):
    """
    Use CNN to predice the number
    """
    
    #prediction:
    pre_label_left=pre_fn(np.reshape(image_left,(-1,1,28,28)))
    pre_label_right=pre_fn(np.reshape(image_right,(-1,1,28,28)))
    
    num_left=[]
    num_left.append(pre_label_left[0][0])
    num_left.append(pre_label_left[0][2])
    return np.array(num_left),pre_label_right[0]
 
    
def narrow_picture(image_list,pixs):
    """
    Narrow the picture:x<28,y<28
    """
    i=0
    for image in image_list:
        img_info=image.shape
        heigh=img_info[0]
        wight=img_info[1]
        heigh=int(heigh/(int(heigh/pixs)+1))
        wight=int(wight/(int(wight/pixs)+1))
        image=cv2.resize(image,(heigh,wight))
        
        image_list[i]=image
        i+=1
   
def Play(flag,loc,m):
    # start = time.perf_counter()  
    if(flag=="right"):
        m.click(loc['mouse_corrcet_x'], loc['mouse_corrcet_y'], 1)
    else:
        m.click(loc['mouse_wrong_x'], loc['mouse_wrong_y'], 1)
           
if __name__ == '__main__':
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
    
    loc_2={"top_x":186,
         "top_y":497,
         "bottom_x":363,
         "bottom_y":660,
         "mouse_corrcet_x":254,
         "mouse_corrcet_y":849,
         "mouse_wrong_x":309,
         "mouse_wrong_y":842,}
    
    #the number of questions
    questions_num=200
    
    #screenshot command
    command = 'import -window root -crop {0}x{1}+{2}+{3} ./screenshot/temp.png'
    command = command.format(loc_2['bottom_x'] - loc_2['top_x'],
                         loc_2['bottom_y'] - loc_2['top_y'],
                         loc_2['top_x'],
                         loc_2['top_y'])
    path="./screenshot/temp.png"
    
    #basis image: "+" and "-":
    sum_path="./data/sum.png"
    sub_path="./screenshot/temp2.png"
    sum_img,sub_img=sum_and_sub(sum_path,sub_path,loc,pixs)

    #click
    m = PyMouse()
    
    """
    CNN initial and load model
    """
    input_var = T.tensor4('inputs')
    network, _ = build_cnn(input_var)
    prediction = lasagne.layers.get_output(network, deterministic=True)
    prediction = T.argmax(prediction, axis=1)
    
    #prediction
    pre_fn = theano.function([input_var], [prediction])
    
    #And load them again later on like this:
    with np.load('mnist_model.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)
    
    """
    Main method
    """
    i=0
    while(i<questions_num):
        i+=1
        time.sleep(0.6)
        start=time.time()
        os.system(command)
        
        #read image
        image_left,image_right,data=contour_split(path,loc)
        
        #narrow picture
        narrow_picture(image_left,pixs)
        narrow_picture(image_right,pixs)
        
        #extend pixs to 28*28
        extend_pixs(image_left,pixs)
        extend_pixs(image_right,pixs)
        image_left=np.float32(np.array(image_left))
        image_right=np.float32(np.array(image_right))
        
        """Prediction"""
        pre_label_left,pre_label_right=CNN_pre(image_left,image_right,pre_fn)  
        
        """Distance"""
        symbol=""
        sum_img=np.float32(np.array(sum_img))
        sub_img=np.float32(np.array(sub_img))
        
        sum_dis=np.linalg.norm(image_left[1]-sum_img)
        sub_dis=np.linalg.norm(image_left[1]-sub_img)
        if sum_dis<sub_dis:
            symbol="+"
        else:
            symbol="-"
            
        """judge the equation"""
        equation_right=""
        for num in pre_label_right:
            equation_right+=str(num)
        equation_right=float(equation_right)
        
        flag=""
        if(symbol=="+"):
            if((pre_label_left[0]+pre_label_left[1])==equation_right):
                flag="right"
                print(str(pre_label_left[0])+"+"+str(pre_label_left[1])+"="+str(equation_right)+":+right")
            else:
                flag="wrong"
                print(str(pre_label_left[0])+"+"+str(pre_label_left[1])+"="+str(equation_right)+":+wrong")
        else:
            if((pre_label_left[0]-pre_label_left[1])==equation_right):
                flag="right"
                print(str(pre_label_left[0])+"-"+str(pre_label_left[1])+"="+str(equation_right)+":-right")
            else:
                flag="wrong"
                print(str(pre_label_left[0])+"-"+str(pre_label_left[1])+"="+str(equation_right)+":-wrong")
        
        if(flag=="right"):
            m.click(loc_2['mouse_corrcet_x'], loc_2['mouse_corrcet_y'], 1)
        else:
            m.click(loc_2['mouse_wrong_x'], loc_2['mouse_wrong_y'], 1)

        print time.time()-start