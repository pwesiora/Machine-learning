import random

import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

FILENAME = "11_8.png"
scaling_factor = 0.33


def count_frames_manual(video):
    total = 0
    while True:
        (grabbed, frame) = video.read()
        if not grabbed:
            break
        total += 1
    return total


img = cv2.imread(FILENAME)
cv2.imshow("Input image", img)
img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
cv2.waitKey()
cv2.destroyAllWindows()
# Zad2
gray_img = cv2.imread(FILENAME, cv2.IMREAD_GRAYSCALE)
gray_img = cv2.resize(gray_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
cv2.imshow("Grayscale", gray_img)
cv2.imwrite("11_2.jpg", gray_img)
cv2.waitKey()
cv2.destroyAllWindows()
# Zad3
# blur_kernel_size = (31, 31)
blur_kernel_size = (15, 15)
gray_img = cv2.imread("11_2.jpg")
gray_blur = cv2.GaussianBlur(gray_img, blur_kernel_size, 0)
cv2.imshow("Grayscale", gray_blur)
cv2.imwrite("11_3.jpg", gray_blur)
cv2.waitKey()
cv2.destroyAllWindows()
# Zad4
# canny_low_threshold = 14
# canny_high_threshold = 120
canny_low_threshold = 40
canny_high_threshold = 100


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


gray_blur = cv2.imread("11_3.jpg")
blur_canny = canny(gray_blur, canny_low_threshold, canny_low_threshold)
cv2.imshow("Canny", blur_canny)
cv2.imwrite("11_4.jpg", blur_canny)
cv2.waitKey()
cv2.destroyAllWindows()

# Extra Zad4 - live video version test for fun
# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#
#     frame = cv2.resize(frame, (600, 460))
#     grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray_blur = cv2.GaussianBlur(grayFrame, blur_kernel_size, 0)
#     video = canny(gray_blur, canny_low_threshold, canny_low_threshold)
#     cv2.imshow("Canny", video)
#     if (cv2.waitKey(1) & 0XFF == ord('q')):
#         break

# Zad5

img = cv2.imread("11_4.jpg")
height, width = img.shape[:2]
print(img.shape[:2])

h = 280
w = 300
x = 0
y = 150
img1 = img[x:x + h, y:y + w]
img2 = np.zeros_like(img)
img2[x:x + h, y:y + w] = img1
cv2.imshow('Grayscale', img2)
cv2.imwrite('11_5.jpg', img2)
cv2.waitKey()
cv2.destroyAllWindows()

# Zad6
src = cv2.imread("11_5.jpg")
dst = cv2.Canny(src, 15, 100, None, 3)
cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
cdstP = np.copy(cdst)
lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
cv2.imshow("Detected lines (in red)", cdstP)
cv2.imwrite("11_6.jpg", cdstP)
cv2.waitKey()
cv2.destroyAllWindows()

# Zad 7

img = cv2.imread("11_2.jpg")
img1 = cv2.imread("11_6.jpg")
img2 = cv2.addWeighted(img, 0.8, img1, 1, 0)
cv2.imshow("sum", img2)
cv2.imwrite("11_7.jpg", img2)
cv2.waitKey()
cv2.destroyAllWindows()

# Zad8
cap = cv2.VideoCapture("vid8.mp4")
canny_low_threshold = 40
canny_high_threshold = 100
while True:
    ret, frame = cap.read()

    frame = cv2.resize(frame, (600, 460))
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(grayFrame, blur_kernel_size, 0)
    video = canny(gray_blur, canny_low_threshold, canny_low_threshold)
    # h = 356
    # w = 634
    # x = 0
    # y = 0
    h = 120
    # w = 300
    w = 360
    x = 230
    # y = 120
    y = 90
    video1 = video[x:x + h, y:y + w]
    video2 = np.zeros_like(video)
    video2[x:x + h, y:y + w] = video1

    # RED LINES

    dst = cv2.Canny(video2, 15, 100, None, 3)
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

    # (356, 634)
    # fps = 30
    # size = (460, 600)
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    # OVERLAP
    cap3 = cv2.addWeighted(frame, 0.8, cdstP, 1, 0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    cv2.putText(cap3, "Piotr Wesiora", (376, 440), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 150, 255), 2, cv2.LINE_4)
    cv2.putText(cap3, "fps:" + str(fps), (430, 30), cv2.FONT_HERSHEY_DUPLEX, 1,
                (0, 150, 255), 2, cv2.LINE_4)
    cv2.imshow("Detected lines (in red)", cap3)
    # videoWriter = cv2.VideoWriter('MyOutput2.avi', fourcc, fps, cap3.shape[:2])
    # videoWriter.write(cap3)

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

print("Total frames counted : {0}".format(length))
cap.release()
cv2.destroyAllWindows()
