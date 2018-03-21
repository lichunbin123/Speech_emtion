import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras import regularizers
import os
import pandas as pd
from sklearn.utils import shuffle
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

global llb
#从本地文件中打开.wav文件
path = 'H:\RawData'
mylist = os.listdir(path)

type(mylist)

#print(mylist[40])
#print(mylist[40][6:-16])

#判断每个音频文件的情感
#参考文档，更具标号设置情感标签

feeling_list=[]
for item in mylist:
    if item[6:-16]=='02' and int(item[18:-4])%2==0:
        feeling_list.append('female_calm')
        
    elif item[6:-16]=='02' and int(item[18:-4])%2==1:
        feeling_list.append('male_calm')
        
    elif item[6:-16]=='03' and int(item[18:-4])%2==0:
        feeling_list.append('female_happy')
        
    elif item[6:-16]=='03' and int(item[18:-4])%2==1:
        feeling_list.append('male_happy')
        
    elif item[6:-16]=='04' and int(item[18:-4])%2==0:
        feeling_list.append('female_sad')
        
    elif item[6:-16]=='04' and int(item[18:-4])%2==1:
        feeling_list.append('male_sad')
        
    elif item[6:-16]=='05' and int(item[18:-4])%2==0:
        feeling_list.append('female_angry')
        
    elif item[6:-16]=='05' and int(item[18:-4])%2==1:
        feeling_list.append('male_angry')
        
    elif item[6:-16]=='06' and int(item[18:-4])%2==0:
        feeling_list.append('female_fearful')
        
    elif item[6:-16]=='06' and int(item[18:-4])%2==1:
        feeling_list.append('male_fearful')
#设置标签        
labels = pd.DataFrame(feeling_list)

#print(labels)

#使用MFCC提取本地音频文件的特征

df = pd.DataFrame(columns=['feature'])
bookmark=0
for index,y in enumerate(mylist):
    if mylist[index][6:-16]!='01' and mylist[index][6:-16]!='07' and mylist[index][6:-16]!='08' and mylist[index][:2]!='su' and mylist[index][:1]!='n' and mylist[index][:1]!='d' and mylist[index][:1]!='A' :
        X, sample_rate = librosa.load('H:\\RawData\\'+y, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, 
                                            sr=sample_rate, 
                                            n_mfcc=13), 
                        axis=0)
        feature = mfccs
        #[float(i) for i in feature]
        #feature1=feature[:135]
        #print(feature.shape)
        df.loc[bookmark] = [feature]
        #print(df.shape)
        bookmark=bookmark+1
        
#print(df)    

#制表
#print('df.shape=',df.shape)
df3 = pd.DataFrame(df['feature'].values.tolist())  
#print('df3.shape=',df3.shape)
#将特征和对应的情感存到同一张表中，情感所在列的列名为‘0’

newdf = pd.concat([df3,labels], axis=1)
#print('newdf.shape=%d',newdf.shape)
rnewdf = newdf.rename(index=str, columns={"0": "label"})

rnewdf = shuffle(newdf)
rnewdf=rnewdf.fillna(0)

#print(rnewdf)

#将表格分为训练集和测试集

newdf1 = np.random.rand(len(rnewdf)) < 0.8
train = rnewdf[newdf1]
test = rnewdf[~newdf1]


#特征值为0到倒数第一列，标签值为最后一列
trainfeatures = train.iloc[:, :-1]
testfeatures = test.iloc[:, :-1]
trainlabel = train.iloc[:, -1:]
testlabel = test.iloc[:, -1:]


X_train = np.array(trainfeatures)
y_train = np.array(trainlabel)
X_test = np.array(testfeatures)
y_test = np.array(testlabel)

lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))
print(type(lb))
lllb=lb