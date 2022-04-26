from django.shortcuts import render
from django.http import HttpResponse
import io
from scipy.io import wavfile as wav
import numpy as np
from .utilitypreproc import MFCCextraction
import tensorflow as tf
import os

# Create your views here.

def home(request):
    return render(request, 'core/home.html')


def process(request):
    if request.method == "POST":
        if request.FILES.get("myAudio", False):
            response = handleUploadFile(request.FILES["myAudio"])
    return HttpResponse(response)

def handleUploadFile(f):

    audio_path = "media/" + f.name + ".wav"
    with open(audio_path, "wb+") as destination:  
        for chunk in f.chunks():  
            destination.write(chunk)

    sr, myrecording = wav.read(audio_path)
    model = "static/net/tcn_004_13_32321616.h5"
    print(model)
    print(myrecording.shape)
    myrecording = np.transpose(myrecording)[0]
    fs = myrecording.shape[0]
    mycut = myrecording
    feat = MFCCextraction([mycut],16000,0.04)
    feat = np.expand_dims(feat,axis=0)[0]
    net = tf.keras.models.load_model(model)
    pred = net.predict(feat)
    print(pred)
    max_id = np.argmax(pred[0])
    commands = ["down","go","left","no","off","on","right","stop","up","yes","not a keyword"]
    word = commands[max_id]
    prob = pred[0][max_id]
    response = str(word)+";"+str(int(round(prob,2)*100))

    os.remove(audio_path)

    return response