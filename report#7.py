

'''
import cv2
import numpy as np

# 이미지 파일 읽기
image = cv2.imread('square.jpg')

# RGB에서 YCbCr로 변환
ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
y_channel = ycbcr[:, :, 0]

# Canny Edge Detection 적용
edges = cv2.Canny(y_channel, 50, 150)

# Hough Line Transform 적용
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# 가장 강한 2개의 직선 선택
strongest_lines = lines[:, 0, :]
strongest_lines = strongest_lines[np.argsort(strongest_lines[:, 0])][-2:]

# 원본 이미지에 직선 그리기
for line in strongest_lines:
    x1, y1, x2, y2 = line
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 결과 이미지 출력
cv2.imshow('Detected Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''


'''
import cv2
import numpy as np

# 비디오 파일 열기
video = cv2.VideoCapture('KakaoTalk_20230523_124248958.mp4')

# 비디오의 프레임 폭과 높이 가져오기
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 비디오 작성자 설정
output_video = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

while True:
    # 비디오에서 프레임 읽기
    ret, frame = video.read()

    if not ret:
        break

    # RGB에서 YCbCr로 변환
    ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y_channel = ycbcr[:, :, 0]

    # Canny Edge Detection 적용
    edges = cv2.Canny(y_channel, 50, 150)

    # Hough Line Transform 적용
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # 가장 강한 2개의 직선 선택
    if lines is not None:
        strongest_lines = lines[:, 0, :]
        strongest_lines = strongest_lines[np.argsort(strongest_lines[:, 0])][-2:]

        # 원본 영상에 직선 그리기
        for line in strongest_lines:
            x1, y1, x2, y2 = line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 결과 비디오에 프레임 쓰기
    output_video.write(frame)

# 비디오 파일 닫기
video.release()
output_video.release()


'''
'''
Canny(src, dst, 50, 200, 3);
cvtColor(dst, cdst, COLOR_GRAY2BGR);
cdstP= cdst.clone();

vector<Vec2f> lines;
HoughLines(dst, lines, 1, CV_PI/180, 150, 0, 0);
for (size_t i=0; i<lines.size(); i++)
{
    float rho= lines[i][0], theta= lines[i][1];
    Point pt1, pt2;
    double a= cos(theta), b= sin(theta);
    double x0=a*rho, y0=b*rho;
    pt1.x= cvRound(x0 + 1000*(-b));
    pt1.y= cvRound(y0 + 1000*(a));
    pt2.x= cvRound(x0 - 1000*(-b));
    pt2.y= cvRound(y0 - 1000*(a));
    line(cdst, pt1, pt2, Scalar(0,0,255), 3, LINE_AA);
}

'''

'''

import cv2
import numpy as np

# 이미지 파일 읽기
image = cv2.imread('400_300.jpg')

# Canny Edge Detection 적용
edges = cv2.Canny(image, 50, 200, 3)

# Gray-scale 이미지를 BGR 형식으로 변환
cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# Hough Line Transform 적용
lines = cv2.HoughLines(edges, 1, np.pi / 180, 150, 0, 0)

if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
        cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

# 결과 이미지 출력
cv2.imshow('Detected Lines', cdst)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''



import cv2
import numpy as np

# 비디오 파일 열기
video = cv2.VideoCapture('KakaoTalk_20230523_124248958.mp4')

while True:
    # 비디오에서 프레임 읽기
    ret, frame = video.read()

    if not ret:
        break

    # Canny Edge Detection 적용
    edges = cv2.Canny(frame, 50, 200, 3)

    # Gray-scale 이미지를 BGR 형식으로 변환
    cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Hough Line Transform 적용
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150, 0, 0)

    if lines is not None:
         # 가장 강한 직선 두 개 선택
        strongest_lines = lines[:, 0, :][np.argsort(lines[:, 0, 1])][-2:]
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
            cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    # 결과 이미지 출력
    cv2.imshow('Detected Lines', cdst)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 파일과 창 닫기
video.release()
cv2.destroyAllWindows()


