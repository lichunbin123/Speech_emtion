from tkinter import *
import tkinter.filedialog
import librosa
from astropy.convolution.boundary_extend import np
from keras.models import load_model
from matplotlib.pyplot import axis
from array import array
import pandas as pd
import os
from sklearn.preprocessing.label import LabelEncoder
from Lb_creator import lb
from tkinter.messagebox import showinfo, showwarning
from PIL import ImageTk
from First_page import *

class MainPage(object):
    
    def __init__(self,master=None):
        self.model = load_model('Detector_Model.h5')
        self.root = master
        print(type(self.root))
        self.page = Frame(self.root)
        self.page.grid(row=1)
        lb0 = Label(self.page,text = '       语音情感识别系统 --MFCC      ',fg='black',compound='center',font=("Arial,12"))
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
                X, sample_rate = librosa.load(filename, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
                sample_rate = np.array(sample_rate)
                #print(type(sample_rate))
                mfccs = np.mean(librosa.feature.mfcc(y=X, 
                                                        sr=sample_rate, 
                                                        n_mfcc=13), 
                                    axis=0)
                feature = mfccs
                #print(feature.shape)
                df = pd.DataFrame(columns=['feature'])
                df.loc[0] = [feature]
                df3 = pd.DataFrame(df['feature'].values.tolist())
                #print(df3,df3.shape)
                
                feature_cnn = np.expand_dims(df3, axis=2)
                #print(feature_cnn.shape)
                pred = self.model.predict(feature_cnn,batch_size = 32,verbose = 1)
                pred1 = pred.argmax(axis=1)
                aaa = pred1.astype(int).flatten()
                predictss = (lb.inverse_transform(aaa))
        
            else:
                
                showwarning('warning','请选择音频文件')
                
        except NameError:
        
            showwarning('warning','请选择文件')
    
        else:
            print(predictss)
            showinfo('The answer of prediction',predictss)