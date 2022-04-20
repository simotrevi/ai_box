# -*- coding: utf-8 -*-
"""
@authors: Anna Dorna, Simone Trevisan
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

def deltaCoef(feat, N):
    NUMFRAMES = len(feat)
    denominator = 2 * sum([i**2 for i in range(1, N+1)])
    delta_feat = np.empty_like(feat)
    padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')   # padded version of feat
    for t in range(NUMFRAMES):
        delta_feat[t] = np.dot(np.arange(-N, N+1), padded[t : t+2*N+1]) / denominator   # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
        
    return delta_feat

#%%
    
def MFCCextraction(data,samplerate,seglen):
    nfft = 1
    while nfft < samplerate*seglen:
        nfft *= 2
    X = [];
    for i in data:
        i = i.astype(np.float)
        if len(i)!=16000:
            i = librosa.resample(i, len(i), 16000, fix=False)
        inputdata = psf.mfcc(signal=i, samplerate=samplerate, winlen=seglen, winstep=seglen/2,
                                          numcep=13, nfilt=26, nfft=nfft, lowfreq=300, highfreq=samplerate/2,
                                          preemph=0.3, ceplifter=0, appendEnergy=False, winfunc=np.hamming)
        X.append(inputdata)    
    return X