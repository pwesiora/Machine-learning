import cv2
import numpy
import imutils
from skimage import io


#ZAD1
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

scaling_factor = 0.25
frame = cv2.imread(r"zad1v5.JPG")
frame = cv2.resize(frame, None, fx =scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

for(x,y,w,h) in face_rects:
    cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

cv2.imshow('unnamed.jpg', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

#ZAD2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

image = cv2.imread(r"zad2v3.JPG")
image = cv2.resize(image, None, fx =scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

gray_filter = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces= face_cascade.detectMultiScale(gray_filter, 7, 4)
face_rects = face_cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)


for(x,y,w,h) in face_rects:
    cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 3)
    roi_gray = gray_filter[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    smile = smile_cascade.detectMultiScale(roi_gray)
    eye = eye_cascade.detectMultiScale(roi_gray)
    for (sx, sy, sw, sh) in smile:
        cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0,255,0), 1)
    for (ex, ey, ew, eh) in eye:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,0,255), 1)

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#ZAD2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

image = cv2.imread(r"zad2v2.JPG")
image = cv2.resize(image, None, fx =scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

gray_filter = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces= face_cascade.detectMultiScale(gray_filter, 7, 4)
face_rects = face_cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)


for(x,y,w,h) in face_rects:
    cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 3)
    roi_gray = gray_filter[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    smile = smile_cascade.detectMultiScale(roi_gray)
    eye = eye_cascade.detectMultiScale(roi_gray)
    for (sx, sy, sw, sh) in smile:
        cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0,255,0), 1)
    for (ex, ey, ew, eh) in eye:
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,0,255), 1)

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(f'Found {len(face_rects)} faces!')



# ZAD3
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cv2.startWindowThread()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    frame = cv2.resize(frame, (800, 560))
    face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

    gray_filter = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for (x, y, w, h) in face_rects:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 3)
    cv2.imshow("Video", frame)
    if(cv2.waitKey(1) & 0XFF==ord('q')):
        break
cap.release()
cv2.destroyAllWindows()

# ZAD4
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cv2.startWindowThread()
cap = cv2.VideoCapture('vid1.mp4')

while True:
    ret, frame = cap.read()

    frame = cv2.resize(frame, (800, 560))
    #frame = cv2.resize(frame, (400, 260))
    gray_filter = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8))
    people_rects = hog.detectMultiScale(frame, winStride=(8,8), padding=(30,30), scale=1.0)
    boxes = numpy.array([[x,y,x+w,y+h] for (x,y,w,h) in boxes])

    for (xa, ya, xb, yb) in boxes:
        cv2.rectangle(frame, (xa, ya), (xb, yb), (0,255, 0), 1)
    for (x, y, w, h) in face_rects:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 1)
    cv2.putText(frame, "People on screen: " + str(len(boxes)), (10, 500), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 150, 255), 3, cv2.LINE_4)
    cv2.imshow("Video", frame)
    if(cv2.waitKey(1) & 0XFF==ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(0)
# print(f'Found {len(boxes)} people!')
# print(f'Found {len(people_rects)} faces!')