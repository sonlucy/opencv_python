"""
정보통신공학과 22012225 손보경
컴퓨터비젼 report#3
"""

import numpy as np
import cv2

imgfile='window.jpeg'
img =cv2.imread(imgfile, cv2.IMREAD_COLOR)
imgtoYCrCb =cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
imgtoBRG =cv2.cvtColor(imgtoYCrCb, cv2.COLOR_YCrCb2BGR)


def PSNR(img, imgtoBGR):  ##주어진 수식 이용해 psnr 사용자 정의 함수 구현
    MSE=np.mean((img-imgtoBGR)**2)
    PSNR= 10*np.log10((255.0*255.0)/MSE)
    return PSNR



print("Opencv함수 이용하여 구한 PSNR :", cv2.PSNR(img, imgtoBRG))
print("수식 이용하여 구한 PSNR :", PSNR(img, imgtoBRG))
