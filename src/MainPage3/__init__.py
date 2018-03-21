from __future__ import print_function
from tkinter import *
import tkinter.filedialog
from astropy.convolution.boundary_extend import np
from keras.models import load_model
from matplotlib.pyplot import axis
from array import array
import pandas as pd
import os
from sklearn.preprocessing.label import LabelEncoder
from tkinter.messagebox import showinfo, showwarning
from PIL import ImageTk
#from First_page import *
import wave
import struct
from scipy import  *
import numpy as np
import scipy.io as scio
import scipy.io.wavfile as wave
from scipy import signal as sig
import os
from keras import backend as K
class MainPage3(object):
    
    def __init__(self,master=None):
        self.model = load_model('Detector_Model_2.h5')
        self.root = master
        print(type(self.root))
        self.page = Frame(self.root)
        self.page.grid(row=1)
        lb0 = Label(self.page,text = '       语音情感识别系统--语谱图       ',fg='black',compound='center',font=("Arial,12"))
        lb0.grid(row=0)  
        lb_s = Label(self.page,text = '-选择单个语音文件识别-',width=20,height=1,fg='blue',compound='center')
        lb_s.grid(row=1) 
        self.lb1 = Label(self.page,text = '',width=60,height=1,fg='green',bg='white')
        self.lb1.grid(row=2,padx=1,column=0)
        
        #该按钮用于选择文件
        btn_opf = Button(self.page,text="选择文件",command=self.xz)
        btn_opf.grid(row=2,sticky=E)
        
        btn_prd = Button(self.page,text='点击预测',command=self.predict,width=60)
        btn_prd.grid(row=3)
        
        btn_back = Button(self.page,text='返回',command=self.back,width=60)
        btn_back.grid(row=4)
        
    def xz(self):
        
        global filename
        filename = tkinter.filedialog.askopenfilename()
        print(filename[-1:])
        
        if filename != '':
            self.lb1.config(text = "文件名："+filename);
        else:
            self.lb1.config(text = "您没有选择任何文件");
    
    def back(self):
        
        self.page.destroy()

    def predict(self):
        
        try:
        #filename如果未定义则会抛出异常
            if(filename[-3:]=='wav'):
                fs,wav_data = wave.read(filename)
                print(fs)
                wav_length = len(wav_data)
                print("length of wave:",wav_length)
                print("data of wav(int):",type(wav_data))
                y=wav_data
                segment_sample_num = 4240
                segment_move = 400
                start = 0
                end = start+segment_sample_num
        
                seg_num = 0
                while end < wav_length :
                    start = start+segment_move
                    end = start+segment_sample_num
                    seg_num += 1
                    
                print(seg_num)
                
                after_segment = [""]*seg_num
                start = 0
                x_test=np.zeros((seg_num,32,129))
                for i in range(seg_num):
                    
                    after_segment[i] = y[start:start+segment_sample_num+1]
                    start =start+segment_move
                    f,t,spectro = sig.spectrogram(after_segment[i], fs, ('hamming'), 256,128,256)
                    #print(f,t)
                    #print(shape(spectro))
                    spectro = log(1+abs(spectro))
                    for a in range(129):
                        for b in range(32):
                            x_test[i][b][a] = spectro[a][b]
                           
                print(shape(x_test))
                if K.image_data_format() == 'channels_first':
                    x_test = x_test.reshape(x_test.shape[0], 1, 32, 129)
                else:
                    x_test = x_test.reshape(x_test.shape[0], 32, 129, 1)

                x_test = x_test.astype('float32')

                print('x_train shape:', x_test.shape)
                print(x_test.shape[0], 'test samples')
                preds = self.model.predict(x_test, batch_size=32, verbose=1)
                print(type(preds))
                print(preds)
                print(shape(preds))
                preds=preds.sum(axis=0)
                print(preds)
                pred_sum = preds.tolist()
                condition = pred_sum.index(max(pred_sum))
                if condition == 0:
                    showinfo('预测', '')
            else:
                
                showwarning('warning','请选择音频文件')
                
        except NameError:
        
            showwarning('warning','请选择文件')
    
        else:
            #print(predictss)
            showinfo('The answer of prediction')