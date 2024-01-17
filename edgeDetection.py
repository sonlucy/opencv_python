'''
import numpy as np
import cv2

imgfile = '400_300.jpg' #
img = cv2.imread(imgfile, cv2.IMREAD_COLOR)   # imgfile을 읽어옴
img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # RGB -> YCrCb

img=cv2.resize(img,(400,300))
cv2.imshow('INPUT IMAGE', img)

Y, Cr, Cb = cv2.split(img_YCrCb)    # Y, Cr, Cb 성분을 분리
Y=cv2.resize(Y,(400,300))
cv2.imshow('GRAY IMAGE', Y)

CEdge = cv2.Canny(Y, 50, 200, 3)
Lines = cv2.HoughLines(CEdge, 1, np.pi/180, 150)

for i in Lines:
    rho, theta = i[0]
    a = np.cos(theta); b = np.sin(theta)
    x0 = a * rho; y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(img, (x1, y1), (x2, y2), (0,255,255), 1)

cv2.imshow('LINE DISPLAY', img)    # RGB 원본(Lines 포함)
CEdge=cv2.resize(CEdge,(400,300))
cv2.imshow('EDGE DISPLAY', CEdge)   # Canny Edge Detection 결과
cv2.waitKey(0)

'''

'''
import numpy as np
import cv2

img=cv2.imread('400_300.jpg') #img 받아오기
img=cv2.resize(img, (400,300)) #사진크기 조정
cv2.imshow('img', img) #원본 img 출력
img_YCbCr=cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) #원본 img를 YCbCr형태로 변환
y, cb, cr=cv2.split(img_YCbCr) #각 원소 분리

for i in range(5):
    y=cv2.blur(y,(3,3)) #edge잡을 때 원하지 않는 부분 검출을 줄이기 위해 blur실행

edge=cv2.Canny(y,10,30) #YCbCr에서 Y성분을 이용하여 edge 검출
edge=np.mat(edge) #HoughLines()함수에서 사용해야하므로 mat형태로 변형
line=cv2.HoughLines(edge,1,np.pi/180,125) #HoughLines()를 이용하여 직선 검출

for i in line: #원본 img에 검출한 직선 성분 긋기
    r,t=i[0]
    a=np.cos(t)
    b=np.sin(t)
    x1=int(a*r+1000*(-b))
    y1=int(b*r+1000*a)
    x2=int(a*r-1000*(-b))
    y2=int(b*r-1000*a)
    cv2.line(img, (x1,y1),(x2,y2),(0,0,255),2)

cv2.imshow('gray',y) #YCbCr에서 Y성분 출력
cv2.imshow('Edge',edge) #검출한 edge 출력
cv2.imshow('line', img) #원본 이미지에 검출한 선을 그은 결과물 출력
cv2.waitKey(0)

'''

'''
컴퓨터비젼 report#7
22012225 손보경

'''

import cv2
import numpy as np

# 비디오 파일 열기
video = cv2.VideoCapture('KakaoTalk_20230523_124248958.mp4')

while True:
    ret, frame = video.read()
    frame = cv2.resize(frame, (400, 300)) #영상이미지 크기 조정
    cv2.imshow('Original', frame)

    YCbCr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb) # 원본-> YCbCr 형태로 변환

    # YCbCr에서 Y 성분 분리
    y, cb, cr = cv2.split(YCbCr)
    for i in range(5):
        y = cv2.blur(y, (3, 3))

    edge = cv2.Canny(y, 10, 30) #y정보 이용해 canny Edge Detection
    edge = np.mat(edge)# Hough Line Transform을 위해 mat 형태로 변환

    # Hough Line Transform을 이용하여 직선 검출
    lines = cv2.HoughLines(edge, 1, np.pi / 180, 120) #적용이미지,거리해상도,각도해상도,스레스홀드값 

    if lines is not None:
        for line in lines:
            r, t = line[0]
            a = np.cos(t)
            b = np.sin(t)
            x1 = int(a * r + 1000 * (-b))
            y1 = int(b * r + 1000 * a)
            x2 = int(a * r - 1000 * (-b))
            y2 = int(b * r - 1000 * a)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('Y(gray)', y)
    cv2.imshow('Edge', edge)
    cv2.imshow('Line', frame)

    if cv2.waitKey(1) == 27:  # esc 누르면 종료
        break


cv2.waitKey(0)
video.release()
cv2.destroyAllWindows()

