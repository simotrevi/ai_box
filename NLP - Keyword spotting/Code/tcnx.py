# -*- coding: utf-8 -*-
"""
@authors: Anna Dorna, Simone Trevisan
"""

import tensorflow as tf
import math
from datagenerator_v2 import load_samples, customGenerator

#%% Load dataset

trainsamples = load_samples("model/trainset.csv")
valsamples = load_samples("model/valset.csv")
testsamples = load_samples("model/testset.csv")

#%% Create datasets generators

batchsize = 100
reduced_flag = False
trainGen = customGenerator(trainsamples,batchsize,shuffle=True,reduced = reduced_flag, training = True)
valGen = customGenerator(valsamples,batchsize,shuffle=False,reduced = reduced_flag)
testGen = customGenerator(testsamples,batchsize,shuffle=False,reduced = reduced_flag)

if reduced_flag:
    trainsize = math.ceil(len(trainsamples)*0.1)
    valsize = math.ceil(len(valsamples)*0.1)
    testsize = math.ceil(len(testsamples)*0.1)
else:
    trainsize = len(trainsamples)
    valsize = len(valsamples)
    testsize = len(testsamples)

batchnum_train = int(math.ceil(trainsize/batchsize))
batchnum_val = int(math.ceil(valsize/batchsize))
batchnum_test = int(math.ceil(testsize/batchsize))


#%% TCN

model_input = tf.keras.layers.Input(shape=(49,13), name="model_input")
conv1 = tf.keras.layers.Conv1D(
    32, 5, strides=1, padding='causal', data_format='channels_last',
    dilation_rate=1, activation='relu',
    kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    kernel_constraint=None, bias_constraint=None)(model_input)
norm1 = tf.keras.layers.LayerNormalization()(conv1)
drop1 = tf.keras.layers.SpatialDropout1D(0.05)(norm1)
conv2 = tf.keras.layers.Conv1D(
    32, 5, strides=1, padding='causal', data_format='channels_last',
    dilation_rate=2, activation='relu',
    kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    kernel_constraint=None, bias_constraint=None)(drop1)
norm2 = tf.keras.layers.LayerNormalization()(conv2)
drop2 = tf.keras.layers.SpatialDropout1D(0.05)(norm2)
conv3 = tf.keras.layers.Conv1D(
    16, 5, strides=1, padding='causal', data_format='channels_last',
    dilation_rate=4, activation='relu',
    kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    kernel_constraint=None, bias_constraint=None)(drop2)
norm3 = tf.keras.layers.LayerNormalization()(conv3)
drop3 = tf.keras.layers.SpatialDropout1D(0.05)(norm3)
conv4 = tf.keras.layers.Conv1D(
    16, 5, strides=1, padding='causal', data_format='channels_last',
    dilation_rate=8, activation='relu',
    kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    kernel_constraint=None, bias_constraint=None)(drop3)
norm4 = tf.keras.layers.LayerNormalization()(conv4)
flatten = tf.keras.layers.Flatten(name='flatten')(norm4)
drop4 = tf.keras.layers.Dropout(0.05)(flatten)
model_output = tf.keras.layers.Dense(11,activation="softmax", name="output")(drop4)

tcn = tf.keras.models.Model(model_input, model_output, name="tcn")
tcn.summary()


#%%


tcn.compile(optimizer= tf.keras.optimizers.Adam(lr=0.0005), loss=tf.keras.losses.CategoricalCrossentropy(),metrics=[tf.keras.metrics.CategoricalAccuracy()])
history = tcn.fit(trainGen, epochs=35,steps_per_epoch = batchnum_train,validation_steps=batchnum_val, validation_data=(valGen),verbose=2)

import pandas as pd
# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history)
hist_json_file = 'history_004_13_32321616.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)


tcn.save('tcn_004_13_32321616.h5') 
 


