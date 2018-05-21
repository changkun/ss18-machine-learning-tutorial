
import h5py

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten, Conv2D, MaxPooling2D
import keras.callbacks

np.random.seed(42)

n_classes = 18

pic_size = 64 # image have size 64x64
c_channels = 3 # color channeld (RGB)

# convolutional layer 1
n_conv_1 = 32
k_size_1 = 3

# convolutional layer 2
n_conv_2 = 64
k_size_2 = 3

# max pool layer
p_size_1 = 2
dropout_1 = .25

# dense layer
n_dense_1 = 128
dropout_2 = .5

def load_data():
    h5f = h5py.File('dataset.h5','r+')
    X_train = h5f['X_train'][:]
    X_test = h5f['X_test'][:]
    h5f.close()    

    h5f = h5py.File('labels.h5','r+')
    y_train = h5f['y_train'][:]
    y_test = h5f['y_test'][:]
    h5f.close()  
    
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.

    return X_train, X_test, y_train, y_test

def net():
    model = Sequential()
    model.add(Conv2D(n_conv_1, kernel_size=(k_size_1, k_size_1), activation='relu', input_shape=(pic_size, pic_size, c_channels)))
    model.add(Conv2D(n_conv_2, kernel_size=(k_size_2, k_size_2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(p_size_1, p_size_1)))
    model.add(Dropout(dropout_1))
    model.add(Flatten())
    model.add(Dense(n_dense_1, activation='relu'))
    model.add(Dropout(dropout_2))
    model.add(Dense(n_classes, activation="softmax"))
    return model

# construct model
model = net()
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = keras.callbacks.History()
checkpoint = keras.callbacks.ModelCheckpoint('weights.hdf5', monitor='val_acc', verbose=0, save_best_only=True, mode='max')

# training
X_train, X_test, y_train, y_test = load_data()
model.fit(X_train,y_train, batch_size=128, epochs=50, verbose=1, validation_data=(X_test, y_test), callbacks=[checkpoint, history])

import json
json.dumps(history, indent=4, separators=(',', ': '))
