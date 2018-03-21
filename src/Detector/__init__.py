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
    '''   
    elif item[:1]=='a':
        feeling_list.append('male_angry')
        
    elif item[:1]=='f':
        feeling_list.append('male_fearful')
        
    elif item[:1]=='h':
        feeling_list.append('male_happy')
        
    #elif item[:1]=='n':
        #feeling_list.append('neutral')
    elif item[:2]=='sa':
        feeling_list.append('male_sad')
 
    '''    
#设置标签        
labels = pd.DataFrame(feeling_list)

print(labels)

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
print('newdf.shape=',newdf.shape)
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

#print(y_train)
print(X_train.shape)


#创建CNN模型

print('Padding sequences')

x_traincnn =np.expand_dims(X_train, axis=2)
x_testcnn= np.expand_dims(X_test, axis=2)

print(x_traincnn.shape)
#模型采用Sequential方法来构造

model = Sequential()

#第一层为卷积层，因为采用音频输入，使用Conv1D,同时将input的参数设为216

model.add(Conv1D(128, 5,padding='same',
                 input_shape=(216,1)))

#第一层的激活函数，在CNN中通常采用relu方式

model.add(Activation('relu'))

#第二层也为卷积层

model.add(Conv1D(128, 5,padding='same'))

#激活函数

model.add(Activation('relu'))

#为输入数据施加Dropout，防止过拟合。 

model.add(Dropout(0.1))

#池化层，压缩长度

model.add(MaxPooling1D(pool_size=(8)))

#卷积层

model.add(Conv1D(128, 5,padding='same',))

#激活

model.add(Activation('relu'))

#卷积层

model.add(Conv1D(128, 5,padding='same',))

#激活

model.add(Activation('relu'))

#卷积层

model.add(Conv1D(128, 5,padding='same',))

#激活

model.add(Activation('relu'))

#防止过拟合

model.add(Dropout(0.2))

#加入卷积层

model.add(Conv1D(128, 5,padding='same',))

#激活

model.add(Activation('relu'))

#使用flatten过度

model.add(Flatten())

#全连接层

model.add(Dense(10))

#使用softmax为激活函数

model.add(Activation('softmax'))

#采用RMS优化器

opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

print('training.....')

cnnhistory=model.fit(x_traincnn, y_train, batch_size=32, epochs=500, validation_data=(x_testcnn, y_test))


#将运行后的模型保存起来

model_name = 'Detector_Model.h5'
#使用方法os.getcwd()来获取当前工作目录，并用join方法得到路径
save_dir = os.path.join(os.getcwd(), 'saved_models')

#保存模型
#第一次运行时创建文件夹，此后不必创建
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
    
model_path = os.path.join(save_dir, model_name)
model.save(model_path)

print('Saved trained model at %s ' % model_path)
print('\n模型训练及保存完成')






      

