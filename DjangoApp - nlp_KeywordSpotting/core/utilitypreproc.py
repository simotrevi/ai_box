# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 18:04:03 2020

@author: simon
"""

import numpy as np
import json
import math
import python_speech_features as psf
import librosa
import tensorflow as tf

#%%

def encode_text(word_to_number, Y_label):
    encoded = [word_to_number[c] for c in Y_label]
    return encoded

#%%

def create_one_hot_matrix(encoded, alphabet_len = 11):
    # Create one hot matrix
    encoded_onehot = np.zeros([len(encoded), alphabet_len])
    tot_chars = len(encoded)
    encoded_onehot[np.arange(tot_chars), encoded] = 1
    return encoded_onehot

#%%

def oneHotEncoding(Y_label, alphabet_len = 11):
    word_to_number = json.load(open('model/word_to_number.json'))
    Y_oh = create_one_hot_matrix(encode_text(word_to_number, Y_label),alphabet_len)
    
    return Y_oh


#%%
    
def MFCCextraction(data,samplerate,seglen):
    #samplerate = 16000
    #seglen = 0.025
    nfft = 1
    while nfft < samplerate*seglen:
        nfft *= 2
    X = [];

    print(data)
    for i in data:
        i = i.astype(np.float)
        if len(i)!=16000:
            i = librosa.resample(i, len(i), 16000, fix=False)
        #i = tf.image.random_crop(i, size=[6240], seed=None, name=None)
        #i = i.numpy()
        inputdata = psf.mfcc(signal=i, samplerate=samplerate, winlen=seglen, winstep=seglen/2,
                                          numcep=13, nfilt=13, nfft=nfft, lowfreq=300, highfreq=samplerate/2,
                                          preemph=0.3, ceplifter=0, appendEnergy=False, winfunc=np.hamming)
#        inputdata = librosa.feature.mfcc(y=i, sr=samplerate, S=None, n_mfcc=2, dct_type=2, norm='ortho', lifter=0,
#                                         n_fft = nfft, hop_length = int(0.02*samplerate), win_length = int(0.04*samplerate),
#                                         window = np.hamming,
#                                         fmin = 300, fmax = samplerate/2, n_mels = 26)
        #print(np.shape(inputdata))
        #delta = deltaCoef(inputdata[:,1:],2)
        #deltadelta = deltaCoef(delta,2)
        #inputdata = np.concatenate((inputdata, delta, deltadelta),axis = 1)
        X.append(inputdata)    
    return X