# -*- coding: utf-8 -*-
"""

@author: 11011105
"""

from turtle import shape
import numpy as np
import cv2


cols=1440#寬
rows=1080#高
testratio=1#讓pattern等比例縮小
cutratio=0.5#將邊緣背景切掉
r=rows*0.03
ratio=np.array([0.2,0.4])#創造圓點的位置,中心為0開始，邊緣為1
d1w=cols*ratio/2#圓點間寬度差距
d1h=rows*ratio/2#圓點間高度差距
im=np.zeros((rows,cols),dtype=np.uint8)
im3=np.zeros((rows,cols),dtype=np.uint8)
im3cut=np.zeros((rows,cols),dtype=np.uint8)
im=cv2.circle(im,(int(cols/2),int(rows/2)),int(r),255,-1)#中心圓點

for i in range(0,len(ratio)):#創造四個方向的點#1.5是配合實驗室實體pattern
    im=cv2.circle(im,(int(cols/2+d1h[i]*1.5),int(rows/2+d1h[i])),int(r),255,-1)
    im=cv2.circle(im,(int(cols/2+d1h[i]*1.5),int(rows/2-d1h[i])),int(r),255,-1)
    im=cv2.circle(im,(int(cols/2-d1h[i]*1.5),int(rows/2+d1h[i])),int(r),255,-1)
    im=cv2.circle(im,(int(cols/2-d1h[i]*1.5),int(rows/2-d1h[i])),int(r),255,-1)
im=255-im#反轉

im_small = cv2.resize(im, (int(cols*testratio), int(rows*testratio)), interpolation=cv2.INTER_AREA)#讓圖片等比例縮小
im3[int(rows/2-rows*testratio/2):int(rows/2+rows*testratio/2),int(cols/2-cols*testratio/2):int(cols/2+cols*testratio/2)]=im_small#將圖片貼到原大小圖片上
im3cut[int(rows/2-rows*cutratio/2):int(rows/2+rows*cutratio/2),int(cols/2-cols*cutratio/2):int(cols/2+cols*cutratio/2)]=im3[int(rows/2-rows*cutratio/2):int(rows/2+rows*cutratio/2),int(cols/2-cols*cutratio/2):int(cols/2+cols*cutratio/2)]#把周圍影像切掉
imtwo=np.zeros((rows,cols*2),dtype=np.uint8)#創建雙眼影像
imtwo[0:rows,0:cols]=im#將pattern影像丟入雙眼image
im3_inv=255-im3
cv2.imwrite("./a85_chart1"+".bmp",im3)
cv2.imwrite("./a85_chart2"+".bmp",im3_inv)
#im=imtwo
#======================圖片變數說明
#im 原始
#im3 等比例縮小後
#im3cut 等比例縮小後再切邊
#imtwo 雙眼pattern
#圖片變數說明=======================
cv2.imwrite("./pattern/chart1"+".bmp",im)#存圖
cv2.imwrite("./pattern/chart1_test"+".bmp",im3)#存圖
im2=np.zeros_like(im)#創造pattern2(全黑影像)
for i in range(0,np.shape(im)[0]):
    for j in range(0,np.shape(im)[1]):
        if i==0 or j==0 or i==np.shape(im)[0]-1 or j==np.shape(im)[1]-1:
            im2[i,j]=255
            
cv2.imwrite("./chart2"+".bmp",im2)#存圖
#==================================for A85存圖
a85_im=np.zeros((rows,cols*2),dtype=np.uint8)
a85_im2=np.zeros((rows,cols*2),dtype=np.uint8)
a85_im3=np.zeros((rows,cols*2),dtype=np.uint8)
a85_im3cut=np.zeros((rows,cols*2),dtype=np.uint8)
a85_im[:,0:cols]=im
a85_im2[:,0:cols]=im2
a85_im3[:,0:cols]=im3
a85_im3cut[:,0:cols]=im3cut
"""
cv2.imwrite("./a85_chart1"+".bmp",a85_im)
cv2.imwrite("./a85_chart2"+".bmp",a85_im2)
cv2.imwrite("./a85_chart1_test"+".bmp",a85_im3)
cv2.imwrite("./a85_chart1_test2"+".bmp",a85_im3cut)
"""
#for A85存圖============================
