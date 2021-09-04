import cv2
import imutils as iu
import numpy as np

img = cv2.imread('image.jpg')
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.imwrite('newimage.jpg', img)
color = (0, 150, 255)
cv2.destroyAllWindows()

# region of interest
roi = img[100:400, 200:650]
cv2.imshow('roi', roi)
cv2.waitKey(0)
cv2.destroyAllWindows()

# resizing
resized = cv2.resize(img, (256, 256))
cv2.imshow('resized', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# resizing2
scale_percent = 20  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
resized2 = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
cv2.imshow("Resized image", resized2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# rotation
rotated = iu.rotate(resized2, 60)
cv2.imshow('rotated', rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

# blur
blurred = cv2.blur(resized2, (5, 5))
cv2.imshow('blurred', blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()

# merge
res = iu.resize(resized2, width=500)
blurres = iu.resize(blurred, width=500)
merged = np.hstack((res, blurres))
cv2.imshow('merged', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# rectangle
img_copy = resized2.copy()
cv2.rectangle(img_copy, (190, 185), (245, 289), color, 2)
cv2.imshow('rectangle', img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()

# line
img2 = np.zeros((256, 256, 3), np.uint8)
cv2.line(img2, (0, 0), (256, 256), color, 4)
cv2.imshow('line', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# polygon
img3 = np.zeros((512, 512, 3), np.uint8)
points = np.array([[40,40], [100, 100], [275, 300], [250, 120], [275, 130]])
cv2.polylines(img3, np.int32([points]), 1, color)
cv2.imshow('polygon', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

# circle
img4 = np.zeros((256, 256, 3), np.uint8)
cv2.circle(img4, (128, 128), 44, color, 4)
cv2.imshow('circle', img4)
cv2.waitKey(0)
cv2.destroyAllWindows()

# text
font = cv2.FONT_HERSHEY_DUPLEX
font2 = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
cv2.putText(resized2, 'Piotrek', (10, 100), font, 4, color, 4, cv2.LINE_4)
cv2.putText(resized2, 'Piotrek', (10, 500), font2, 4, color, 3, cv2.LINE_4)
cv2.imshow('text', resized2)
cv2.waitKey(0)
cv2.destroyAllWindows()
