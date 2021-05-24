import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout, Activation 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras import backend as K
from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import numpy as np

labels = ['Lalettan', 'Mamukka']
img_width, img_height = 150, 150
train_data_dir = 'Data/train'
val_data_dir = 'Data/validation'
train = 1200
test = 150
epochs = 100
batch_size = 20
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3) #150, 150, 3

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'binary'
)

val_generator = test_datagen.flow_from_directory(
    val_data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'binary'
)

#creating neural network
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.summary()

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))         
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

model.compile(loss = 'binary_crossentropy',
            optimizer = 'rmsprop',
            metrics = ['accuracy'])

#Augmentation configuration for training

model.fit(
    train_generator,
    steps_per_epoch = 2635 // batch_size,
    epochs = epochs,
    validation_data = val_generator,
    validation_steps = 370 // batch_size
)

model.save('model.h5')
