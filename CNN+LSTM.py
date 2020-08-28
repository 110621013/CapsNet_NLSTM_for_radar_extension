# -*- coding: utf-8 -*-
import os
os.chdir('C:/Users/user/Desktop/all_conbine')
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import numpy as np
import keras
from keras import models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding,TimeDistributed,Conv2D,Input, Flatten,Reshape ,MaxPooling2D, Dropout,Activation,BatchNormalization
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
from nested_lstm import NestedLSTM
import os
import argparse
from keras import callbacks, metrics
#from keras.preprocessing.image import ImageDataGenerator
import time
import pandas as pd
#import glob
from sklearn.preprocessing import MinMaxScaler
#from keras.utils import multi_gpu_model

from PIL import Image
from radar_data_config import *
dataset = 'radar_data'
button = 'test' #train/test

scaler = MinMaxScaler(feature_range=(0, 1))
rows = width
cols = length
sample_num = train_num
timesteps = 15
pre_steps = 10
#pre_steps is the prediction time.It can be 1 or 5 or 10
link_num = point_num
test_sample_num = test_num
data_source_path = 'C:/Users/user/Desktop/all_conbine/DATA/'+dataset+'/'


trainX = np.load(data_source_path+'train_img_array.npy')
trainX = trainX.reshape(sample_num,rows,cols,1)
X_train=[]
for i in range(sample_num-timesteps-pre_steps+1):
    X_train.append(trainX[i:i+timesteps])
x_train = np.asarray(X_train,dtype='float32')
print ('x_train.shape',x_train.shape)

y1 = np.load(data_source_path+'train_point_value_array.npy')
y1 = np.asarray(y1).astype('float32')
y_train = []
for i in range(sample_num-timesteps-pre_steps+1):
    y_train.append(y1[i+timesteps:i+timesteps+pre_steps])
y_train = np.asarray(y_train).astype('float32')
y_train = y_train.reshape(sample_num -timesteps -pre_steps +1, link_num*pre_steps)
print ('y_train.shape',y_train.shape)


testX = np.load(data_source_path+'test_img_array.npy')
testX = testX.reshape(test_sample_num,rows,cols,1)
X_test=[]
for i in range(test_sample_num-timesteps-pre_steps+ 1):
    X_test.append(testX[i:i+timesteps])
x_test = np.asarray(X_test,dtype='float32')
print ('x_test.shape',x_test.shape)

y_1 = np.load(data_source_path+'test_point_value_array.npy')
y_1 = np.asarray(y_1).astype('float32')
y_test = []
for i in range(test_sample_num-timesteps-pre_steps+ 1):
    y_test.append(y_1[i+timesteps:i+timesteps+pre_steps])
y_test = np.asarray(y_test).astype('float32')
y_test = y_test.reshape(test_sample_num-timesteps-pre_steps+1, link_num*pre_steps)
print ('y_test.shape',y_test.shape)

def mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), 1, np.inf))
    return 100. * K.mean(diff, axis=-1)

if __name__ == '__main__':
    if button == 'train':
        print('Build model...')
        model = Sequential()
        model.add(TimeDistributed(Conv2D(16, (3, 3), padding='same'), input_shape=((timesteps,rows,cols,1))))
        model.add(Activation('relu'))
        model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2),padding='same')))

        model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same')))
        model.add(Activation('relu'))
        model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2),padding='same')))

        model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same')))
        model.add(Activation('relu'))
        model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2),padding='same')))

        model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same')))
        model.add(Activation('relu'))
        model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding='same')))

        model.add(TimeDistributed(Flatten()))

        model.add(LSTM(800,return_sequences=True))
        model.add(LSTM(800))
        model.add(Dropout(0.2))
        model.add(Dense(link_num*pre_steps))
        model.add(Activation('linear'))
        model.summary()

        model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss='mse',metrics=[mean_absolute_percentage_error])

        print('Train...')
        model.fit(x_train, y_train,
                batch_size =32,
                epochs=20, validation_data=(x_test, y_test),
                callbacks=[ModelCheckpoint('./h5/cnn+lstm_{}_{}.h5'.format(str(timesteps), str(pre_steps)), monitor='val_loss' , save_best_only=True, save_weights_only=False,verbose=1)])

        print('evaluation.....')
        score = model.evaluate(x_test, y_test)
        print('Test score:', score)

        print('prediction.....')
        predictions = model.predict(x_test[:168])
        print("predictions shape:", predictions.shape)

    elif button == 'test':
        model = keras.models.load_model('./h5/cnn+lstm_{}_{}.h5'.format(str(timesteps), str(pre_steps)))
        model.summary()
        score = model.evaluate(x_test, y_test)
        print('Test score:', score)

        print('prediction.....')
        predictions = model.predict(x_test)
        print("predictions shape:", predictions.shape, y_test.shape)

        x_lim, y_lim = 10, 16
        array = np.empty((y_lim, x_lim), dtype='float32')
        for i in range(pre_steps): #10
            for x in range(x_lim):
                for y in range(y_lim):
                    array[y, x] = y_test[0, i*160 + x*16 + y]
            img = Image.fromarray(array)
            img.convert('L').save('./test/cnn+lstm_{}_{}_{}.png'.format(str(timesteps), str(pre_steps), str(i)))

        for i in range(pre_steps):
            for x in range(x_lim):
                for y in range(y_lim):
                    array[y, x] = predictions[0, i*160 + x*16 + y]
            img = Image.fromarray(array)
            img.convert('L').save('./predictions/cnn+lstm_{}_{}_{}.png'.format(str(timesteps), str(pre_steps), str(i)))


