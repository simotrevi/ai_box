# -*- coding: utf-8 -*-

import sounddevice as sd
import soundfile as sf
from tkinter import *
import tensorflow as tf
from utilitypreproc import MFCCextraction
import numpy as np
from scipy.io import wavfile as wav
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

def Voice_rec():
    fs = 16000
    # seconds
    duration = 1.4
    myrecording = sd.rec(int(duration * fs), 
                         samplerate=fs, channels=1)
    sd.wait()

    return sf.write('my_Audio_file.wav', myrecording, fs)

def Process():
    samplepath = "my_Audio_file.wav"
    myrecording = wav.read(samplepath)[1]
    model = "trained_models/dscnn_004_13_32_no.h5"
    reshape = True
    if v.get()==1:
        model = "trained_models/dscnn_004_13_32_no.h5"
    else: 
        model = "trained_models/tcn_004_13_32321616.h5"
        reshape = False
    print(model)
    fs=16000
    mycut = myrecording[int(fs*0.2):int(fs*1.2)]
    feat = MFCCextraction([mycut],16000,0.04)
    
    if reshape:
        feat = np.expand_dims(feat, axis=3)
    else:
        feat = np.expand_dims(feat,axis=0)[0]
    net = tf.keras.models.load_model(model)
    pred = net.predict(feat)
    print(pred)
    commands = ["down","go","left","no","off","on","right","stop","up","yes","filler"]
    x = np.arange(len(commands))
    fig = Figure(figsize = (5,5),dpi=100)
    plot1= fig.add_subplot(111)
    plot1.bar(x,pred[0])
    plot1.set_xticks(x)
    plot1.set_xticklabels(commands)
    plot1.set_ylabel('Probability')
    plot1.set_title('Output of the network')
    canvas = FigureCanvasTkAgg(fig,master = master)
    canvas.draw()
    canvas.get_tk_widget().grid(row=20,column=2, columnspan=2, rowspan=5)
    toolbar = NavigationToolbar2Tk(canvas,
                                   master)
    toolbar.update()
    canvas.get_tk_widget().grid(row=30,column=2, columnspan=2, rowspan=5)
    return
  
master = Tk()
master.title("Keyword Spotting Demo")
master.geometry("700x700")
v = IntVar()
a = StringVar()
v.set(1)

Label(master, text=" Model : "
     ).grid(row=0, sticky=W, rowspan=5)

Radiobutton(master, 
               text="DS-CNN",
               padx = 20, 
               variable=v, 
               value=1).grid(row=0,column=2, columnspan=2, rowspan=5)

Radiobutton(master, 
               text="TCN",
               padx = 20, 
               variable=v, 
               value=2).grid(row=0,column=4, columnspan=2, rowspan=5)
 
Label(master, text=" Voice Recoder : "
     ).grid(row=5, sticky=W, rowspan=5)

  
b = Button(master, text="Rec", command=Voice_rec)
b.grid(row=5, column=2, columnspan=2, rowspan=2,
       padx=5, pady=5)

c = Button(master, text="Process", command=Process)
c.grid(row=5, column=4, columnspan=2, rowspan=2,
       padx=5, pady=5)




mainloop()