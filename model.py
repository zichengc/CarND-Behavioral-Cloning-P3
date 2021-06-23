import numpy as np
import csv
from pathlib import  Path
import tensorflow as tf
import math
import cv2
# from os import listdir
import os
import random
import sklearn
from keras.models import Sequential
from keras.layers import (Dense, Flatten, Conv2D, Lambda, Cropping2D, Dropout)
from sklearn.model_selection import train_test_split

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            # batch_labels = labels[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0]
                image = cv2.imread('../data/' + name)
                images.append(image)
                angles.append(batch_sample[1])

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield (X_train, y_train)

def main():
    ### collect image files and steering angles from log file
    samples = []
    correct = 0.25
    with open('../data/driving_log.csv') as f:
        reader = csv.reader(f)
        ### skip header
        next(reader)
        direc = [0,1,-1]
        for line in reader:
            for i in range(3):
                filename = line[i].strip()
                steering = float(line[3]) + direc[i]*correct
                samples.append([filename,steering])

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    # Set  batch size
    batch_size=32

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)            
            


    ### build model based on NVIDIA end-to-end self driving car paper
    img_shape = (160,320,3)
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=img_shape))
    model.add(Cropping2D(cropping = ((50,20),(0,0))))
    model.add(Conv2D(24,(5,5), activation= 'relu', strides= (2,2)))
    model.add(Conv2D(36,(5,5), activation= 'relu', strides= (2,2)))
    model.add(Conv2D(48,(5,5), activation= 'relu', strides= (2,2)))
    model.add(Conv2D(64,(3,3), activation= 'relu'))
    model.add(Conv2D(64,(3,3), activation= 'relu'))
    model.add(Flatten())
    model.add(Dropout(0.2))  
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(optimizer = 'adam',
                 loss = 'mse')

    img_shape = (160,320,3)
    # model = bcnn(img_shape)
    model.summary()

    model.fit_generator(train_generator,
                        steps_per_epoch=math.ceil(len(train_samples)/batch_size),
                        validation_data=validation_generator,
                        validation_steps=math.ceil(len(validation_samples)/batch_size),
                        epochs=5, verbose=1)
    model.save('model.h5')

if __name__ == '__main__':
    main()