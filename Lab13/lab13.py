import matplotlib
from keras.models import load_model
from tensorflow.keras.datasets import mnist
import os

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

index = np.random.choice(np.arange(len(X_train)), 24, replace=False)
figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(14, 9))
for item in zip(axes.ravel(), X_train[index], y_train[index]):
    axes, image, target = item
    axes.imshow(image, cmap=plt.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(target)
    plt.tight_layout()
plt.show()
print("RESHAPING")
X_train = X_train.reshape((60000, 28, 28, 1))
print(X_train.shape)
X_test = X_test.reshape((10000, 28, 28, 1))
print(X_test.shape)

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
print(y_train.shape)
print(y_train[0])

y_test = to_categorical(y_test)
print(y_test.shape)

from tensorflow.keras.models import Sequential

# cnn = Sequential()
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

# cnn.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
# cnn.add(MaxPooling2D(pool_size=(2, 2)))
# cnn.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
# cnn.add(MaxPooling2D(pool_size=(2, 2)))
# cnn.add(Flatten())
# cnn.add(Dense(units=128, activation='relu'))
# cnn.add(Dense(units=10, activation='softmax'))
# cnn.summary()

from tensorflow.keras.utils import plot_model
from IPython.display import Image
import pydot

# plot_model(cnn, to_file='convnet.png', show_shapes=True, show_layer_names=True)
# Image(filename='convnet.png')
# cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# cnn.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
#
# save_dir = "/results/"
# model_name = 'keras_mnist.h5'
# model_path = os.path.join(save_dir, model_name)
# cnn.save(model_path)
# print('Saved trained model at %s ' % model_path)


cnn = load_model('keras_mnist.h5')

loss, accuracy = cnn.evaluate(X_test, y_test)
print("loss: " + str(loss))
print("accuracy: " + str(accuracy))
przypuszczenia = cnn.predict(X_test)
# print(y_test)
# print(y_test.shape)
# print(type(y_test))
# print(type(y_test[0]))
#
# print(przypuszczenia)
# print(przypuszczenia.shape)
# print(type(przypuszczenia))
# print(type(przypuszczenia[0]))

from numpy import *
for indeks, przypuszczenie in enumerate(przypuszczenia[0]):
    print(f'{indeks}: {przypuszczenie:.10%}')

obrazy =  X_test.reshape((10000, 28, 28))
chybione_prognozy = []

for i, (p, e) in enumerate(zip(przypuszczenia, y_test)):

    prognozowany, spodziewany = np.argmax(p), np.argmax(e)

    if prognozowany != spodziewany:
        chybione_prognozy.append((i, obrazy[i], prognozowany, spodziewany))
print(len(chybione_prognozy))

figure2, axes2 = plt.subplots(nrows=4, ncols=6, figsize=(8, 6))

for axes2, element in zip(axes2.ravel(), chybione_prognozy):
    indeks, obraz, prognozowany, spodziewany = element
    axes2.imshow(obraz, cmap=plt.cm.gray_r)
    axes2.set_xticks([])
    axes2.set_yticks([])
    axes2.set_title(target)
    axes2.set_title( f'indeks: {indeks}\np: {prognozowany}; s: {spodziewany}')
    plt.tight_layout()
plt.show()

import imageio
im1 = imageio.imread("im1.png")
im2 = imageio.imread("im2.png")
im3 = imageio.imread("im3.png")

gray1 = np.dot(im1[...,:3], [0.299, 0.587, 0.114])
gray2 = np.dot(im2[...,:3], [0.299, 0.587, 0.114])
gray3 = np.dot(im3[...,:3], [0.299, 0.587, 0.114])
plt.imshow(gray1, cmap = plt.get_cmap('gray'))
plt.show()
print(gray1.shape)
print(gray2.shape)
print(gray3.shape)
# gray1.resize(900,1000)
# gray2.resize(700,600)
# gray3.resize(700,600)

gray1 = gray1.reshape(1, 28, 28, 1)
gray2 = gray2.reshape(1, 28, 28, 1)
gray3 = gray3.reshape(1, 28, 28, 1)

# X_test2 = (gray1, gray2, gray3)

przypuszczenia2 = cnn.predict(gray1)
print(przypuszczenia2.argmax())
przypuszczenia2 = cnn.predict(gray3)
print(przypuszczenia2.argmax())
