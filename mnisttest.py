import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.datasets import mnist


(X_train_image, y_train_label),(X_test_image, y_test_label) = mnist.load_data()

from tensorflow.keras.utils import to_categorical 
X_train = X_train_image.reshape(60000, 28, 28, 1)
X_test = X_test_image.reshape(10000, 28, 28, 1)
y_train = to_categorical(y_train_label, 10)
y_test = to_categorical(y_test_label, 10)

from keras import models
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
model = models.Sequential()
model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size=128)