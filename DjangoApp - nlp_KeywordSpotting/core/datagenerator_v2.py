# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 13:16:16 2020

@author: simon
"""

 
import pandas as pd 
import random
import math
from scipy.io import wavfile as wav
from utilitypreproc import MFCCextraction, create_one_hot_matrix
import tensorflow as tf
import numpy as np
import librosa


def load_samples(csv_path):
    data = pd.read_csv(csv_path,usecols=['file_name','label','class_name'])
    file_names = list(data.iloc[:,0])
    labels = list(data.iloc[:,1])
    samples=[]
    for samp,lab in zip(file_names,labels):
        samples.append([samp,lab])
    return samples

def manipulate(data, noise_factor, sampling_rate, shift_max):
    #SHIFT
    shift = np.random.randint(sampling_rate * shift_max)
    direction = np.random.randint(0, 2)
    if direction == 1:
        shift = -shift
    shifted_data = np.roll(data, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        shifted_data[:shift] = 0
    else:
        shifted_data[shift:] = 0
    #NOISE
    noise = np.random.randn(len(shifted_data))
    noise_shifted_data = shifted_data + noise_factor * noise
    # Cast back to same data type
    noise_shifted_data = noise_shifted_data.astype(type(data[0]))
    #PITCH
    pitch_factor = np.random.randint(-2,3)
    pitch_noise_shift_data = librosa.effects.pitch_shift(noise_shifted_data, sampling_rate, pitch_factor)
    return pitch_noise_shift_data

def customGenerator(samples, batch_size, shuffle=False, reduced = False, training = False):
    if reduced:
        samples = samples[0:int(len(samples)*0.1)]
    num_samples = len(samples)
    num_batches = math.ceil(num_samples/batch_size)
    
    while True:
        if shuffle:
            random.shuffle(samples)
        
        batch_index = 0
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            batch_index = batch_index + 1
            rawdata = []
            rawdata_aug = []
            paths = [item[0] for item in batch_samples]
            labels = [item[1] for item in batch_samples]
            for s in paths:
                sound = wav.read(s)[1]
                sr = wav.read(s)[0]
                rawdata.append(sound)
                if training:
                    rawdata_aug.append(manipulate(np.array(sound,dtype=float),0.1,sr,0.1))
            datafeat = MFCCextraction(rawdata,16000,0.04)
            #datafeat = [s.reshape(49,13,1) for s in datafeat]
            labelsonehot = create_one_hot_matrix(labels)
            #datafeat = [tf.transpose(s) for s in datafeat]
            datafeat = tf.convert_to_tensor(datafeat)
            if training:
                datafeat_aug = MFCCextraction(rawdata_aug,16000,0.02)
                #datafeat_aug = [tf.transpose(s) for s in datafeat_aug]
                datafeat_aug = tf.convert_to_tensor(datafeat_aug)
            else: datafeat_aug = datafeat

                
            yield  (datafeat_aug, labelsonehot)
            
                
        
