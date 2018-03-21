import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
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
from keras.models import load_model 
import tkinter  
from  tkinter import ttk  #导入内部包  
from tkinter.messagebox import showinfo

model = load_model('Detector_Model.h5')

path = 'H:\RawData'
mylist = os.listdir(path)

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


labels = pd.DataFrame(feeling_list)



df = pd.DataFrame(columns=['feature'])
bookmark=0
for index,y in enumerate(mylist):
    if mylist[index][6:-16]!='01' and mylist[index][6:-16]!='07' and mylist[index][6:-16]!='08' and mylist[index][:2]!='su' and mylist[index][:1]!='n' and mylist[index][:1]!='d' and mylist[index][:1]!='A' :
        X, sample_rate = librosa.load(path+'\\'+y, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, 
                                            sr=sample_rate, 
                                            n_mfcc=13), 
                        axis=0)
        feature = mfccs
        #[float(i) for i in feature]
        #feature1=feature[:135]
        df.loc[bookmark] = [feature]
        bookmark=bookmark+1


df3 = pd.DataFrame(df['feature'].values.tolist())  

#将特征和对应的情感存到同一张表中，情感所在列的列名为‘0’

newdf = pd.concat([df3,labels], axis=1)
rnewdf = newdf.rename(index=str, columns={"0": "label"})

rnewdf = shuffle(newdf)
rnewdf=rnewdf.fillna(0)

#print(rnewdf)

#将表格分为训练集和测试集

newdf1 = np.random.rand(len(rnewdf)) < 0.2
train = rnewdf[newdf1]
test = rnewdf[~newdf1]


#特征值为0到倒数第一列，标签值为最后一列
testfeatures = test.iloc[:, :-1]

testlabel = test.iloc[:, -1:]

X_test = np.array(testfeatures)
y_test = np.array(testlabel)

lb = LabelEncoder()

y_test = np_utils.to_categorical(lb.fit_transform(y_test))

#print(y_train)

#创建CNN模型

print('提取测试集...')

x_testcnn= np.expand_dims(X_test, axis=2)

print(x_testcnn)

print('测试...')
preds = model.predict(x_testcnn, 
                         batch_size=32, 
                         verbose=1)

preds1=preds.argmax(axis=1)

abc = preds1.astype(int).flatten()

predictions = (lb.inverse_transform((abc)))

preddf = pd.DataFrame({'predicted_values': predictions})
actual=y_test.argmax(axis=1)
abc123 = actual.astype(int).flatten()

#print(abc)

actualvalues = (lb.inverse_transform((abc123)))
print(type(actualvalues))

actualdf = pd.DataFrame({'actual_values': actualvalues})
print(type(actualdf))
finaldf = actualdf.join(preddf)
finaldf.to_csv('H:\\Predictions.csv', index=False)
print('\n\n输出预测值与实际值的对比表格：\n\n')
final_sum = finaldf.groupby('actual_values').count().join(finaldf.groupby('predicted_values').count())
print(finaldf.groupby('actual_values').count().join(finaldf.groupby('predicted_values').count()))