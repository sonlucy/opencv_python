
"""
imgfile= 'window.jpeg'
img=cv2.imread(imgfile, cv2.IMREAD_COLOR)

cv2.imshow('img',img)
cv2.waitey(0)

"""
"""

#구현 1
imgfile='window.jepg'
img=cv2.imread(imgfile,cv2.IMREAD_COLOR)
img=cv2.resize(img, (300,200))
cv2.imshow('Original', img)
img=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb) 
cv2.imwrite('cvtcolor.jpeg',img)           
cv2.imshow('1. BGR2YCrCb',img[:,:,0])

imgfile='cvtcolor.jpeg'                  
img=cv2.imread(imgfile,cv2.IMREAD_COLOR)  
img=cv2.cvtColor(img,cv2.COLOR_YCrCb2BGR) 
cv2.imshow('1. YCrCb2BGR',img)

     """
"""
정보통신공학과 22012225 손보경
"""

import numpy as np
import cv2

#구현 2
imgfile='window.jpeg'

img=cv2.imread(imgfile,cv2.IMREAD_COLOR)
cv2.imshow('img', img)

red, green, blue = cv2.split(img)      

#RGB->YCbCr
Y= 0.257 * red + 0.504 * green + 0.098 * blue + 16
Cb= -0.148 * red - 0.291 * green + 0.439 * blue + 128
Cr= 0.439 * red - 0.368 * green - 0.071 * blue + 128

img[:,:,0] = Y; img[:,:,1] = Cr; img[:,:,2] = Cb
YCbCr=np.uint8(img)
cv2.imshow('YCbCr', YCbCr)

#YCbCr->RGB
R = 1.164*(Y-16) + 1.596 * (Cr-128)
G = 1.164*(Y-16) - 0.813 * (Cr-128) - 0.391 * (Cb-128)
B = 1.164*(Y-16) + 2.018 * (Cb-128)

img[:,:,0] = R; img[:,:,1] = G; img[:,:,2] = B
cv2.imshow('RGB',img)

cv2.waitKey(0)

























