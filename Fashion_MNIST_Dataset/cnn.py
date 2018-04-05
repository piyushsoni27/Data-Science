#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 03:17:40 2018

@author: piyush
"""

import mnist_reader

import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D        ##Conv2D --> for images
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

xtrain, ytrain = mnist_reader.load_mnist('Data', kind='train')
xtest, ytest = mnist_reader.load_mnist('Data', kind='t10k')

"""
plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
plt.imshow(xtrain[0].reshape(28,28), cmap='gray')
plt.title("Ground Truth : {}".format(ytrain[0]))

plt.subplot(121)
plt.imshow(xtest[0].reshape(28,28), cmap='gray')
plt.title("Ground Truth : {}".format(ytest[0]))
"""

xtrain = xtrain.reshape(-1, 28,28,1).astype("float32")
xtest = xtest.reshape(-1, 28,28,1).astype("float32")

xtrain = xtrain/255
xtest = xtest/255

ytrain_one_hot = to_categorical(ytrain)
ytest_one_hot = to_categorical(ytest)

xtrain,xvalid,train_label,valid_label = train_test_split(xtrain, ytrain_one_hot, test_size=0.2, random_state=13)


batch_size = 64
epochs = 20
num_classes = 10

fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(Dense(num_classes, activation='softmax'))


fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

print(fashion_model.summary())


fashion_train = fashion_model.fit(xtrain, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(xvalid, valid_label))

test_eval = fashion_model.evaluate(xtest, ytest_one_hot, verbose=0)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

#Plot the accuracy and loss plots between training and validation data:
accuracy = fashion_train.history['acc']
val_accuracy = fashion_train.history['val_acc']
loss = fashion_train.history['loss']
val_loss = fashion_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


