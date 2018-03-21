from __future__ import print_function
import wave
import struct
from scipy import  *
import numpy as np
import scipy.io as scio
import scipy.io.wavfile as wave
from scipy import signal as sig
import os

path = 'H:\RawData2'
mylist = os.listdir(path)
mark=0
for index,y_name in enumerate(mylist):
    fs,wav_data = wave.read(path+'\\' +y_name,mmap=False)
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
    y_test=np.zeros((seg_num,7))
    #print(mylist[index][5])       
    for i in range(seg_num):
        if mylist[index][5]=='F':
            y_test[i][0]=1
        if mylist[index][5]=='N':
            y_test[i][1]=1
        if mylist[index][5]=='W':
            y_test[i][2]=1
        if mylist[index][5]=='T':
            y_test[i][3]=1
        if mylist[index][5]=='A':
            y_test[i][4]=1
        if mylist[index][5]=='L':
            y_test[i][5]=1
        if mylist[index][5]=='E':
            y_test[i][6]=1
        after_segment[i] = y[start:start+segment_sample_num+1]
        start =start+segment_move
        f,t,spectro = sig.spectrogram(after_segment[i], fs, ('hamming'), 256,128,256)
        print(f,t)
        #print(shape(spectro))
        spectro = log(1+abs(spectro))
        
        for a in range(129):
            for b in range(32):
                x_test[i][b][a] = spectro[a][b]
    last_x = x_test
    last_y = y_test
    if(mark==0):
        sum_x = last_x
        sum_y = last_y
        mark=1
    else:
        sum_x = np.append(sum_x, last_x, axis=0)
        sum_y = np.append(sum_y, last_y, axis=0)
    #print(y_name)
    #print(type(sum))           
    #print(sum.shape[0])
    #print(x_test[90])
scio.savemat('H:\\NewData.mat',{'X':sum_x,'Y':sum_y})
#f=scio.loadmat('H:\\NewData.mat')
#y_train = f['Y']
#print(shape(y_train))
#print(y_train)
print('finish')