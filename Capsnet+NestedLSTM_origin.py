# -*- coding: utf-8 -*-

from __future__ import print_function
import os
os.chdir('C:/Users/user/Desktop/all_conbine')
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import numpy as np
import keras
from keras import models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding,TimeDistributed,Conv2D,Input, Flatten,Reshape ,MaxPooling2D, Dropout,Activation, LSTM
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
from nested_lstm import NestedLSTM
import os
import argparse
from keras import callbacks
#from keras.preprocessing.image import ImageDataGenerator
import time
import pandas as pd
#import glob
from sklearn.preprocessing import MinMaxScaler
#from keras.utils import multi_gpu_model

from PIL import Image
from origin_radar_data_config import *
dataset = 'radar_data'
button = 'train' #train/test

scaler = MinMaxScaler(feature_range=(0, 1))
rows = width
cols = length
sample_num = train_num
timesteps = 15 #time lags
pre_steps = 10
#pre_steps is the prediction time.It can be 1 or 5 or 10
link_num = width*length
test_sample_num = test_num
data_source_path = 'C:/Users/user/Desktop/all_conbine/DATA/'+dataset+'/'

trainX = np.load(data_source_path+'train_img_array_origin.npy')
trainX = trainX.reshape(sample_num,rows,cols,1)
X_train=[]
for i in range(sample_num-timesteps-pre_steps+1):
    X_train.append(trainX[i:i+timesteps])
x_train = np.asarray(X_train,dtype='float32')
print ('x_train.shape',x_train.shape)

y1 = np.load(data_source_path+'train_point_value_array_origin.npy')
y1 = np.asarray(y1).astype('float32')
y_train = []
for i in range(sample_num-timesteps-pre_steps+1):
    y_train.append(y1[i+timesteps:i+timesteps+pre_steps])
y_train = np.asarray(y_train).astype('float32')
print ('y_train.shape',y_train.shape)
y_train = y_train.reshape(sample_num -timesteps -pre_steps +1, link_num*pre_steps)
print ('y_train.shape',y_train.shape)


testX = np.load(data_source_path+'test_img_array_origin.npy')
testX = testX.reshape(test_sample_num,rows,cols,1)
X_test=[]
for i in range(test_sample_num-timesteps-pre_steps+ 1):
    X_test.append(testX[i:i+timesteps])
x_test = np.asarray(X_test,dtype='float32')
print ('x_test.shape',x_test.shape)

y_1 = np.load(data_source_path+'test_point_value_array_origin.npy')
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

def CapsNet(input_shape, n_class, routings):
    x = Input(shape=input_shape)

    conv1 =Conv2D(filters=128, kernel_size=9, strides=2, padding='valid', activation='relu', name='conv1')(x)

    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=16, kernel_size=9, strides=4, padding='valid')

    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps')(primarycaps)

    out_caps = Flatten()(digitcaps)
    # outputs = Dense(278)(out_caps)

    # Models for training and evaluation (prediction)
    train_model = models.Model(x,  out_caps)

    return train_model


if __name__ == '__main__':
    if button == 'train':
        print('Build model...')
        capsnet = CapsNet(input_shape=[rows,cols,1],n_class=30,routings=3)
        capsnet.summary()

        model = Sequential()
        model.add(TimeDistributed(capsnet, input_shape = (timesteps,rows,cols,1)))
        model.add(NestedLSTM(800, depth=2))
        model.add(Dropout(0.2))
        model.add(Dense(link_num*pre_steps))
        model.summary()

        model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss='mse',metrics = [mean_absolute_percentage_error])

        print('Train...')
        model.fit(x_train, y_train,
                batch_size =32,
                epochs=20, validation_data=(x_test, y_test),
                callbacks=[ModelCheckpoint('./h5/capsnet+nlstm_origin_{}_{}.h5'.format(str(timesteps), str(pre_steps)), monitor='val_loss' , save_best_only=True, save_weights_only=False,verbose=1)])

        print('evaluation.....')
        score = model.evaluate(x_test, y_test)
        print('Test score:', score)

        print('prediction.....')
        predictions = model.predict(x_test[:168])
        print("predictions shape:", predictions.shape)

    elif button == 'test':
        model = keras.models.load_model('./h5/capsnet+nlstm_origin_{}_{}.h5'.format(str(timesteps), str(pre_steps)), custom_objects={'CapsuleLayer': CapsuleLayer, 'NestedLSTM': NestedLSTM})
        model.summary()
        score = model.evaluate(x_test, y_test)
        print('Test score:', score)

        print('prediction.....')
        predictions = model.predict(x_test)
        print("predictions shape:", predictions.shape, y_test.shape)

        x_lim, y_lim = 70, 70
        array = np.empty((y_lim, x_lim), dtype='float32')
        for i in range(pre_steps): #10
            for x in range(x_lim):
                for y in range(y_lim):
                    array[y, x] = y_test[0, i*x_lim*y_lim + x*y_lim + y]
            img = Image.fromarray(array)
            img.convert('L').save('./test/capsnet+nlstm_origin_{}_{}_{}.png'.format(str(timesteps), str(pre_steps), str(i)))

        for i in range(pre_steps):
            for x in range(x_lim):
                for y in range(y_lim):
                    array[y, x] = predictions[0, i*x_lim*y_lim + x*y_lim + y]
            img = Image.fromarray(array)
            img.convert('L').save('./predictions/capsnet+nlstm_origin_{}_{}_{}.png'.format(str(timesteps), str(pre_steps), str(i)))

