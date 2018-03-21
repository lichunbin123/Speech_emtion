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
#from Lb_creator import lb
from tkinter.messagebox import showinfo, showwarning
from PIL import ImageTk
from First_page import *
from keras.models import load_model
from sklearn.utils import shuffle
from keras.utils import np_utils


class MainPage2(object):
    
    def __init__(self,master=None):
        self.model = load_model('Detector_Model.h5')
        self.root = master
        print(type(self.root))
        self.page = Frame(self.root)
        self.page.grid(row=1)
        self.path = StringVar()
        lb0 = Label(self.page,text = '语音情感识别系统',width=49,fg='black',compound='center',font=("Arial,12"))
        lb0.grid(row=0)  
        lb_s = Label(self.page,text = '-输入语音文件夹路径进行识别-',width=30,height=1,fg='blue',compound='center')
        lb_s.grid(row=1) 
        
        self.lb2 = Entry(self.page,textvariable=self.path,width=60)
        self.path.set('请在此处输入文件路径')
        self.lb2.grid(row=2)
        
        btn_prd = Button(self.page,text='预测并将结果打印到表格',command=self.predict,width=60)
        btn_prd.grid(row=3)
        
        btn_back = Button(self.page,text='返回',command=self.back,width=60)
        btn_back.grid(row=4)
    
    def back(self):
        
        self.page.destroy()

    def predict(self):
        
        try:
        #filename如果未定义则会抛出异常
            path = self.path.get()
            mylist = os.listdir(path)

            feeling_list=[]
            for item in mylist:
                if   item[6:-16]=='02' and int(item[18:-4])%2==0:
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
            #showinfo('提示', '提取测试集')
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
            #showinfo('提示', '正在测试...')
            preds = self.model.predict(x_testcnn, 
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
    
            actualdf = pd.DataFrame({'actual_values': actualvalues})
            
            finaldf = actualdf.join(preddf)
            
            finaldf.to_csv('H:\\预测实际对照表.csv', index=False)
            showinfo("提示","表格打印完成，已保存到H盘目录下")
            print('\n\n输出预测值与实际值的对比表格：\n\n')
            
            print(finaldf.groupby('actual_values').count().join(finaldf.groupby('predicted_values').count()))
            
            #showinfo("预测值与实际值的对比", finaldf.groupby('actual_values').count())    
        except FileNotFoundError:
        
            showwarning('warning','该路径不存在，请重新输入')