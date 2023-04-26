# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:07:01 2020

@author: 10601021
"""
# =============================================================================
# The program evaluates flare effect based on stadardized specification
# 18844, in which a normal exposured image (IM1, with 225 grey level), a plane
# black image (IM2) captured with the same exposure value as IM1 and an over-
# exposured (normally 8x) image are fed.
# A map (cnt_mtx) is generated and a result image (rgb_img) is saved to the 
# root file.
# =============================================================================
import numpy as np
#import easygui
import matplotlib.pyplot as plt
import cv2
from numpy.lib.function_base import average
#import blm_filter_img_0812_A85R as blm
RATIO_E1_E3 = 8
RATIO_CONTOUR_REGION = 50/70
NUM_H, NUM_W = 5, 5
VISUALIZATION_ON = False


#需要更改的參數===================
contour_x_limit = [0,4000]#圓的x軸位置(下限,上限)
contour_y_limit = [0,3000]#圓的y軸位置(下限,上限)
contoursArea_limit = [10000,290000]#圓點的面積(下限,上限)
THRESHOLD = 0.15#閥值(不太需要改)
max_contour_num=9#圓點數量，可以設1、5、9
#需要更改的參數===================

def find_avg_intensity_in_contour(img, cnt):
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    tmp_mask = np.zeros_like(img)
    r = (cv2.contourArea(cnt) / np.pi)**0.5 * RATIO_CONTOUR_REGION
    tmp_mask = cv2.circle(tmp_mask,(cX,cY),int(r),255,-1)
    return np.sum(img[tmp_mask>0]) / (np.pi*r**2), tmp_mask

def find_avg_intensity_around_contour(img, cnt):
# =============================================================================
# The function calculates average intensity around a found contour in a 
# given image
# img: input image
# cnt: contour array
# return: float 
# =============================================================================
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    r = (cv2.contourArea(cnt) / np.pi)**0.5
    d = int((1 + 1) * r)
    tmp_mask = np.zeros_like(img)
    tmp_mask = cv2.circle(tmp_mask,(cX - d,cY),int(r * RATIO_CONTOUR_REGION),255,-1)
    tmp_mask = cv2.circle(tmp_mask,(cX + d,cY),int(r * RATIO_CONTOUR_REGION),255,-1)
    tmp_mask = cv2.circle(tmp_mask,(cX,cY - d),int(r * RATIO_CONTOUR_REGION),255,-1)
    tmp_mask = cv2.circle(tmp_mask,(cX,cY + d),int(r * RATIO_CONTOUR_REGION),255,-1)
    # imm = (tmp_mask.astype(np.float) + img.astype(np.float))
    # plt.figure()
    # plt.imshow(imm)
    return np.sum(img[tmp_mask>0]) / (np.pi*(r* RATIO_CONTOUR_REGION)**2) / 4, tmp_mask
def butter2d_lp( shape, f, n, pxd=1):
    """Designs an n-th order lowpass 2D Butterworth filter with cutoff
    frequency f. pxd defines the number of pixels per unit of frequency (e.g.,
    degrees of visual angle)."""
    pxd = float(pxd)
    rows, cols = shape
    x = np.linspace(-0.5, 0.5, cols) * cols / pxd
    y = np.linspace(-0.5, 0.5, rows) * rows / pxd
    radius = np.sqrt((x ** 2)[np.newaxis] + (y ** 2)[:, np.newaxis])
    filt = 1 / (1.0 + (radius / f) ** (2 * n))
    return filt
def butter2d_bp( shape, cutin, cutoff, n, pxd=1):
    """Designs an n-th order bandpass 2D Butterworth filter with cutin and
    cutoff frequencies. pxd defines the number of pixels per unit of frequency
    (e.g., degrees of visual angle)."""
    return butter2d_lp(shape, cutoff, n, pxd) - butter2d_lp(shape, cutin, n, pxd)

#img_file = easygui.fileopenbox()

def clc_blm(img_file,lowf,highf):
    print (img_file)
    orig_image = cv2.imread(img_file, 0)
    orig_image2 = orig_image.copy() # bakcup the original image
    shape = orig_image.shape
    rows, cols = shape
    # lowf = 0.05
    # highf = 0.8
    cutin = lowf * (cols / 2)
    cutoff = highf * (cols / 2)
    import time
    print ("!!!!!!!!!!!!!!!!START FILTERING!!!!!!!!!!!!!!!!!!!!!!!!!!")
    strtime = time.time()
    # orig_image = orig_image.astype(np.float32)
    # orig_image = (orig_image / np.max(orig_image))**4
    fft_orig = np.fft.fftshift(np.fft.fft2(orig_image))



    filt = butter2d_bp(orig_image.shape, cutin, cutoff,2, pxd=1)
    '''A85Q_0420'''
    # filt[300:800,500:3500] = 0
    # filt[3000-800:3000-300,500:3500] = 0
    # filt[750:2250,400:1100] = 0
    # filt[750:2250,4000-1100:4000-400] = 0
    '''high f A85Q Kowa + 12MP_0416'''
    """
    filt[1450:1550,850:1050] = 0
    filt[1450:1550,4000-1050:4000-850] = 0
    filt[300:800,1940:2080] = 0
    filt[3000-800:3000-300,1940:2080] = 0
    filt[1450:1550,500:600] = 0
    filt[1450:1550,4000-600:4000-500] = 0
    filt[1920:2020,1900:2100] = 0
    filt[3000-2020:3000-1920,1900:2100] = 0
    """
    # filt = cv2.ellipse(filt, (2000,1500), (1100,800), 0, 0, 360, 0, 150)
    # filt = cv2.ellipse(filt, (2000,1500), (1500,1200), 0, 0, 360, 0, 150)
    #================test bwbp==============
    #plt.figure()
    #lowf = 0.0082
    #highf = 0.06
    #lowf2 = 0.2
    #highf2 = 0.4
    #cutin = lowf * (cols / 2)
    #cutoff = highf * (cols / 2)
    #cutin2 = lowf2 * (cols / 2)
    #cutoff2 = highf2 * (cols / 2)
    #for i in range(2,6):
    # filttest = butter2d_bp(orig_image.shape, cutin, cutoff, i, pxd=1)
    # filttest2 = butter2d_bp(orig_image.shape, cutin2, cutoff2, i, pxd=1)
    # filttestsum = 0.5*(filttest+filttest2)
    # plt.plot(filttest[len(filttest)//2,len(filttest[0])//2:len(filttest[0])-1])
    # =======================================

    fft_new = fft_orig * filt * -1
    new_image = np.fft.ifft2(np.fft.ifftshift(fft_new))
    endtime = time.time()
    print ("TIME = " + str(endtime - strtime))
    new_image = np.real(new_image)
    # plt.figure()
    # plt.imshow(new_image)
    # for i in range(0,4000,100):
    # for j in range(0,3000,100):
    # if i + 100 < 4000 and j + 100 < 3000:
    # new_image[j:j+100, i: i +100] = new_image[j:j+100, i: i +100] / np.mean(new_image[j:j+100, i: i +100])
    # Norm_img = new_image
    Norm_img = new_image / np.mean(orig_image)
    print ("Normalizing facotr: " + str(np.mean(orig_image[orig_image.shape[1]*2//5:orig_image.shape[1]*3//5,orig_image.shape[0]*2//5:orig_image.shape[0]*3//5])))
    print ("Max = " +str(Norm_img.max()))
    plt.figure()
    plt.imshow(Norm_img)




    return Norm_img

#======================圖檔位置======================
chart1_1x_file = "./chart1_1.bmp"#flare pattern 1倍曝光
chart2_1x_file = "./chart2_1.bmp"#black pattern 1倍曝光
chart1_8x_file = "./chart1_8.bmp"#flare pattern 8倍曝光
#======================圖檔位置======================

# chart1_1x_file,chart2_1x_file,chart1_8x_file = easygui.fileopenbox(multiple=True,filetypes= "*.png")

imWhite_1x = cv2.imread(chart1_1x_file,0)
imBlack_1x = cv2.imread(chart2_1x_file,0)
imWhite_8x = cv2.imread(chart1_8x_file,0)

filt_img =clc_blm(chart1_1x_file,0.01,0.2)
thr_tst = np.zeros_like(imWhite_1x)
thr_tst[filt_img>THRESHOLD] = 255
"""
kernel_size=401
imWhite_1x = cv2.GaussianBlur(imWhite_1x,(kernel_size, kernel_size), 0)
imBlack_1x = cv2.GaussianBlur(imBlack_1x,(kernel_size, kernel_size), 0)
imWhite_8x = cv2.GaussianBlur(imWhite_8x,(kernel_size, kernel_size), 0)
imWhite_1x = cv2.GaussianBlur(imWhite_1x,(kernel_size, kernel_size), 0)
imBlack_1x = cv2.GaussianBlur(imBlack_1x,(kernel_size, kernel_size), 0)
imWhite_8x = cv2.GaussianBlur(imWhite_8x,(kernel_size, kernel_size), 0)
imWhite_1x = cv2.GaussianBlur(imWhite_1x,(kernel_size, kernel_size), 0)
imBlack_1x = cv2.GaussianBlur(imBlack_1x,(kernel_size, kernel_size), 0)
imWhite_8x = cv2.GaussianBlur(imWhite_8x,(kernel_size, kernel_size), 0)
"""
#ret, thr_imWhite_1x = cv2.threshold(imWhite_1x,THRESHOLD,255,cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thr_tst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


cnt_list = []

cnt_center_list = np.array((0,0))
rgb_img = cv2.imread(chart1_1x_file,1)
rgb_img2= cv2.imread(chart2_1x_file,1)
rgb_img3= cv2.imread(chart1_8x_file,1)
rgb_img4=np.copy(rgb_img)
mask = np.zeros_like(imWhite_1x)
### Check performance of coontour-finding process
### TODO: Exception for contours noise within
def areasort(elem):
    
    return cv2.contourArea(elem)
for c in contours:
    print(cv2.contourArea(c))
    
    if cv2.contourArea(c) > contoursArea_limit[0] and cv2.contourArea(c) < contoursArea_limit[1] and c[0][0][1] > contour_y_limit[0] and c[0][0][1] < contour_y_limit[1] and c[0][0][0] > contour_x_limit[0] and c[0][0][0] < contour_x_limit[1]: # c[0][0][0]>150 -> temporal solution
        cv2.drawContours(mask, [c], -1, 255, -1)
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # cnt_center_list.append(np.array(cX,cY))
        cnt_center_list = np.c_[cnt_center_list,np.array((cX,cY))]
        cnt_list.append(c)
cnt_list.sort(key=areasort)
cnt_center_list =cnt_center_list.transpose((1,0))[1::,:]

# plt.figure()
# plt.imshow(mask)
cv2.namedWindow('My Image', cv2.WINDOW_NORMAL)
cv2.imshow('My Image',mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
### Calculate average intensity within each contour
num=0
result=[]
watch_yb3=[]
watch_yb2=[]
watch_yw1=[]
cnt_info_list = []
R1=0
R2=0
R3=0
R4=0
R5=0
R6=0
R7=0
R8=0
R9=0
if max_contour_num == 1:
    flare_calibration=[R5]
elif max_contour_num == 5:
    flare_calibration=[R7,R6,R5,R4,R3]
elif max_contour_num == 9:
    flare_calibration=[R9,R8,R7,R6,R5,R4,R3,R2,R1]
else :
    flare_calibration=np.zeros(max_contour_num)


for i in range(0,min(len(cnt_list),max_contour_num)):
    M = cv2.moments(cnt_list[i])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    yb3,_ = find_avg_intensity_in_contour(imWhite_8x,cnt_list[i])#找8倍曝光FLARE PATTERN圓點區域灰階值
    yb2, mask_b = find_avg_intensity_in_contour(imBlack_1x,cnt_list[i])#找1倍曝光黑色PATTERN 圓點區域灰階值
    yw1, mask_w = find_avg_intensity_around_contour(imWhite_1x,cnt_list[i])#一倍曝光FLARE PATTERN 圓點外圍四個點區域
    f = (yb3 / RATIO_E1_E3 - yb2) / yw1 * 100#計算FLARE 值
    f=f-flare_calibration[num]
    result.append(f)
    watch_yb3.append(yb3)
    watch_yb2.append(yb2)
    watch_yw1.append(yw1)
    cnt_info_list.append([cX,cY,f])
    #rgb_img[mask_b>0,2] = 255
    num+=1
    rgb_img3[_>0] = (255,0,0)
    rgb_img2[mask_b>0] = (255,0,0)
    rgb_img4[mask_w>0] = (255,0,0)
    cv2.putText(rgb_img, str(np.round(f,2))+"%", (cnt_list[i][0][0][0],cnt_list[i][0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5 , cv2.LINE_AA)#顯示FLARE值在圖片上
avgresult=average(np.array(result)) 
yb3_show=average(np.array(watch_yb3))#9個點取平均秀出來
yb2_show=average(np.array(watch_yb2))#9個點取平均秀出來
yw1_show=average(np.array(watch_yw1))#9個點取平均秀出來
print(num)
if num!=max_contour_num:
    print("!!!!!detect error!!!!!")
print("flare:"+str(round(avgresult,3)))
print("yb3(8倍曝光FLARE PATTERN圓點區域):"+str(round(yb3_show,3)))
print("yb2(1倍曝光黑色PATTERN 圓點區域):"+str(round(yb2_show,3)))
print("yw1(一倍曝光FLARE PATTERN 圓點外圍四個點區域 ):"+str(round(yw1_show,3)))
# plt.figure()
# plt.imshow(rgb_img)

fname = 'Result_' + chart1_1x_file.split('\\')[-1]
# cv2.imwrite('Result_999_3_0bx_cross.png', rgb_img)
cv2.imwrite(fname.split('.')[0]+"_"+str(round(avgresult,3))+'.jpg', rgb_img)
cv2.imwrite('chart2x1.jpg', rgb_img2)
cv2.imwrite('chart1x8.jpg', rgb_img3)
cv2.imwrite('chart1x1.jpg', rgb_img4)
"""
### sorting
cnt_mtx = np.zeros((NUM_H,NUM_H))
for i in range(0,NUM_H):
    tmp_arr = np.array(cnt_info_list[i*NUM_H:(i+1)*NUM_H])

    arg = np.argsort((tmp_arr[:,0]))
    # print (arg)
    # print (tmp_arr)
    c = 0
    for idx in arg:
        cnt_mtx[NUM_H - i -1,c] = tmp_arr[idx,2]
        c += 1
print (np.mean(cnt_mtx))
plt.figure()
plt.imshow(cnt_mtx)
### fitting and visualization
"""
if VISUALIZATION_ON:
    def func(data, a, b, c, d, e, f):
        return a + data[:,0]*b + data[:,1]*c + d*data[:,0]*data[:,1] + e*data[:,0]**2 + f* data[:,1]**2
    import scipy
    from scipy import signal
    import scipy.optimize as optimize
    
    X,Y = np.meshgrid(np.arange(500, 3000, 50), np.arange(200, 2600, 50))
    XX = X.flatten()
    YY = Y.flatten()
    data2 = np.array(cnt_info_list)
    guess = (10,0,0,0,0,0)
    params, pcov = optimize.curve_fit(func, data2[:,:2], data2[:,2], guess)
    # ZZ = func(data2, params[0], params[1], params[2], params[3], params[4], params[5])
    dot = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2],params).reshape(X.shape)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax.plot_surface(X, Y, dot, rstride=1, cstride=1, alpha=0.5)
    ax.scatter(data2[:,0], data2[:,1], data2[:,2], c='g', s=100)
    # ax.plot_surface(data2[:,0], data2[:,1], data2[:,2], rstride=1, cstride=1, alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_zlabel('flare (%)')
    
    ###  x,y coordinate to polar coordinate
    cnt_info_list_r = np.array((0,0))
    for cnt_info in cnt_info_list:
        r = ((cnt_info[0] - 2000)**2 + (cnt_info[1] - 1500)**2 )**0.5
        cnt_info_list_r = np.c_[cnt_info_list_r,np.array((r,cnt_info[2]))]
    cnt_info_list_r = cnt_info_list_r.transpose((1,0))[1::,:]
    p = np.poly1d(np.polyfit(cnt_info_list_r[:,0], cnt_info_list_r[:,1], 3))
    t = np.linspace(0, 1500, 1500)
    plt.figure()
    plt.plot(cnt_info_list_r[:,0], cnt_info_list_r[:,1], 'o')
    plt.plot(t, p(t), 'g-')
    
    X,Y = np.meshgrid(np.arange(0, NUM_H), np.arange(0, NUM_W))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, cnt_mtx , rstride=1, cstride=1, alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_zlabel('flare (%)')
    ax.set_zlim3d(5,25)
    # plt.plot(cnt_info_list_r_999_3[:,0], cnt_info_list_r_999_3[:,1], 'o')
    # plt.plot(t, p_999_3(t), 'b-')