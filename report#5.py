'''
컴퓨터비젼 report#5
22012225 손보경
'''

from operator import contains
import numpy as np
import cv2

img=cv2.imread('400_300.jpg') #원본이미지 불러오기
cv2.imshow('original_img',img) #원본이미지 출력


############################### 구현1 ####################
img_YCbCr1=cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) #원본이미지 YCbCr로 변환
y, cb,cr=cv2.split(img_YCbCr1) #YCbCr 색상 분리 후
Y=cv2.equalizeHist(y) #Y정보만 equalization
img_YCbCr1 = cv2.merge([Y, cb, cr]) #위에서 histogram equalization한 Y값과 원래 Cb, Cr 값을 합병
img_result1=cv2.cvtColor(img_YCbCr1, cv2.COLOR_YCrCb2BGR) #YCbCr -> RGB로 복원
cv2.imshow('HistogramEq_img1',img_result1) 


psnr1=cv2.PSNR(img, img_result1) #opencv함수 사용한 결과 PSNR
print("구현1 PSNR = ", psnr1)

############################### 구현2 ####################
l=400
r=300
num= np.zeros(256) # 0 ~ 256
for i in range(0,r): 
  for j in range(0,l):
      b = y[i,j] #밝기 값
      num[b] += 1 #밝기 별 빈도 수 
      
Sum=np.zeros(256)  
Sum[0]=num[0]
for i in range(0,255):
    Sum[i+1]=Sum[i]+num[i+1] #sum of num
    
sum=np.zeros(256)
c=np.zeros(256)
max=255
num_pixels=l*r #전체 픽셀 수

for i in range(0,256): #Method 1 사용
    sum[i]=((max+1)/num_pixels)*Sum[i] -1 
    c[i]=np.ceil(sum[i]) #올림연산
for i in range(0,r):
    for j in range(0,l):
        y[i,j]= c[y[i,j]] # histogram equalization
        
img_YCbCr2 = cv2.merge([y,cb,cr]) #위에서 histogram equalization한 y값과 원래 Cb, Cr 값을 합병
img_result2 = cv2.cvtColor(img_YCbCr2, cv2.COLOR_YCrCb2BGR) #YCbCr -> RGB로 복원
cv2.imshow('HistogramEq_img2', img_result2)


psnr2=cv2.PSNR(img, img_result2) #수식 사용한 결과 PSNR
print("구현2 PSNR = ", psnr2)

cv2.waitKey(0)
