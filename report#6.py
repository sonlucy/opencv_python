'''
컴퓨터비젼 report#6
정보통신공학과 22012225 손보경
'''

import numpy as np
import cv2

imgfile = 'window.jpeg'
img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
imgtoYCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

Y, Cr, Cb = cv2.split(imgtoYCrCb)


for i in range(5):
    Y = cv2.blur(Y, (3, 3)) #3x3필터 사용, Y영상을 입력 영상으로 필터 적용



img_YCrCb_Filter = cv2.merge((Y, Cr, Cb)) #Averaging filtering한 Y정보와 Cr Cb 합병
img_RGB_Filter = cv2.cvtColor(img_YCrCb_Filter, cv2.COLOR_YCrCb2BGR)


cv2.imshow('Origin', cv2.resize(img, (400,300)))
cv2.imshow('Filtering', cv2.resize(img_RGB_Filter, (400,300)))

print("PSNR = ", cv2.PSNR(img, img_RGB_Filter)) #opencv함수 이용해 PSNR 출력

cv2.waitKey(0)
