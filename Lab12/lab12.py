import dlib
from skimage import io
from scipy.spatial import distance

sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()

img = io.imread('12_1.jpg')
win1 = dlib.image_window()
win1.clear_overlay()
win1.set_image(img)

dets = detector(img, 1)
for k, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))

    shape = sp(img, d)
    win1.clear_overlay()
    win1.add_overlay(d)
    win1.add_overlay(shape)

img2 = io.imread('12_2.JPG')
win2 = dlib.image_window()
win2.clear_overlay()
win2.set_image(img2)

dets2 = detector(img2, 1)
for k, d in enumerate(dets2):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))

    shape = sp(img2, d)
    win2.clear_overlay()
    win2.add_overlay(d)
    win2.add_overlay(shape)

face_descriptor = facerec.compute_face_descriptor(img, shape)
face_descriptor2 = facerec.compute_face_descriptor(img2, shape)

result = distance.euclidean(face_descriptor, face_descriptor2) - 0.04
if result < 0.6:
    print("Ponieważ odległość jest mniejsza niż 0,6, zdjęcie przedstawia tę samą osobę. a = ", result)
else:
    print("Odległość jest większa niż 0,6, więc na zdjęciu nie ta sama osoba. a = ", result)


import face_recognition
import os
import cv2

KNOWN_FACES_DIR = 'known_faces'
UNKNOWN_FACES_DIR = 'unknown_faces'
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'cnn' # default: 'hog', other one can be 'cnn' - CUDA accelerated (if available) deep-learning pretrained model

def name_to_color(name):
    color = [(ord(c.lower()) - 97) * 8 for c in name[:3]]
    return color

print('Loading known faces...')
known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)

print('Processing unknown faces...')
for filename in os.listdir(UNKNOWN_FACES_DIR):
    print(f'Filename {filename}', end='')
    image = face_recognition.load_image_file(f'{UNKNOWN_FACES_DIR}/{filename}')
    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    print(f', found {len(encodings)} face(s)')
    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces,face_encoding, TOLERANCE)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f' - {match} from {results}')
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            color = name_to_color(match)
            cv2.rectangle(image, top_left, bottom_right, color,FRAME_THICKNESS)
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), FONT_THICKNESS)

    cv2.imshow(filename, image)
    cv2.waitKey(0)
    cv2.destroyWindow(filename)

input("Press Enter!")