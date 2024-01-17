"""
정보통신공학과 22012225 손보경
컴퓨터비젼 report#2
구현2
"""

import numpy as np
import cv2

imgfile= 'window.jpeg'
img=cv2.imread(imgfile, cv2.IMREAD_COLOR)
cv2.imshow('img', img)

frame=np.array([[.299, .587, .114],[-.172, -.339, .511],[.511, -.428, -.083]])
YCbCr=img.dot(frame.T)
YCbCr[:,:,[1,2]]+=128
YCbCr=np.uint8(YCbCr)

cv2.imshow('YCbCr', YCbCr)


frame=np.array([[1, 0, 1.371], [1, -.226, -.698], [1, 1.732, 0]])
rgb= YCbCr.astype(np.float64)
rgb[:,:,[1,2]]-=128
rgb=rgb.dot(frame.T)
np.putmask(rgb, rgb>255, 255)
np.putmask(rgb, rgb<0, 0)
rgb= np.uint8(rgb)  

cv2.imshow('RGB', rgb)

cv2.waitKey(0)
