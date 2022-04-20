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
    
def filterBank(nfilt, nfft, samplerate, fmin):
    fmax = samplerate/2
    melmax = 2595 * np.log10(1+fmax/700.)
    melmin = 2595 * np.log10(1+fmin/700.)
    
    melcent = np.linspace(melmin,melmax,nfilt+2)
    
    bins = np.floor((nfft+1)*(700*(10**(melcent/2595.0)-1))/samplerate)
    
    fbank = np.zeros([nfilt,nfft//2+1])
    for j in range(0,nfilt):
        for i in range(int(bins[j]), int(bins[j+1])):
            fbank[j,i] = (i - bins[j]) / (bins[j+1]-bins[j])
        for i in range(int(bins[j+1]), int(bins[j+2])):
            fbank[j,i] = (bins[j+2]-i) / (bins[j+2]-bins[j+1])
            
    return fbank

#%%
    
def framesig(sig, frame_len, frame_step):
    slen = len(sig)
    if not (slen == 16000):
        sig = np.concatenate((sig,np.zeros((16000 - slen,))))
        slen = len(sig)
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0*slen - frame_len) / frame_step))
    
    padlen = int((numframes - 1) * frame_step + frame_len) #zeros to add at the end
    zeros = np.zeros((padlen - slen,))
    padsignal = np.concatenate((sig, zeros))
    
    frames = []
    for i in range(numframes):
        temp = np.zeros(frame_len)
        for j in range(frame_len):
            temp[j] = padsignal[j+(frame_step*i-1)]
        frames.append(temp)
    
    return frames

#%%
    
def powerSpect(segments, nfft):
    
    fft_spectrum = np.fft.rfft(segments, nfft)
    
    return 1.0 / nfft * np.square(np.absolute(fft_spectrum))

#%%
    
def logMelSpect(fbank,spect):
    
    melspect = np.dot(spect,fbank.T)
    melspect = np.where(melspect == 0,np.finfo(float).eps,melspect)
    logmelspect = np.log(melspect)
    
    return logmelspect

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
    #samplerate = 16000
    #seglen = 0.025
    nfft = 1
    while nfft < samplerate*seglen:
        nfft *= 2
    X = [];

    
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