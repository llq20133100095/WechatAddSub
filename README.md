WechatAddSub
===

一、基础说明
---
    1.本代码主要实现了微信“加减大师”的辅助程序。
    2.本代码主要运行在ubuntu上和python2.7中。
    3.首先把手机和电脑链接，利用chrome的Vysor插件对手机进行投屏。
    4.然后利用python中的“os.system（command）”对手机显示进行截屏。
    5.获取截屏中数字的具体位置坐标。（利用ubuntu中GIMP可以获取截图的坐标）
    6.利用opencv读取截图，进行灰度化和二值化。
    7.识别数字的算法：利用theano和lasagne框架构建了CNN模型。训练集是mnist+加减大师中的数字。识别率能达到很高。
    
    缺点：
    1.本代码运行只能做到200题，超出200题的数字没有识别到。并且200题之后速度跟不上“加减大师”的时间条。
    
二、运行说明
---
    1.手机投屏到电脑上，测出数字的坐标：
    
    
    测出“对”和“错”的坐标位置。把这些坐标填入main.py的loc_2中，并填入需要做的题目数量questions_num，for example：
```Java
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
```
    
    2.运行WechatJiaJian中的main.py
