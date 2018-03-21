from tkinter import *
from tkinter.messagebox import showinfo, showwarning
import tkinter.filedialog
from PIL import ImageTk
from bokeh.layouts import column
from MainPage import MainPage
from MainPage2 import MainPage2
from MainPage3 import MainPage3
'''
root = Tk()
root.title('123')
img = ImageTk.PhotoImage(file='python.jpg')
lb_image = Label(root,image=img,text='123')
lb_image.pack()
'''
#第一个界面
class First_page(object):
    
    def __init__(self,master = None):
        self.root = master
        self.creat()
        
    def creat(self):
        self.page = Frame(self.root)
        self.page.grid(row=1)
        lb1=Label(self.page,text='       语音情感识别系统        ',font=("Arial,12"),fg='blue')
        lb1.grid(row=1)
        lb2=Label(self.page,text='---请选择识别方式---',fg='green')
        lb2.grid(row=2)
        btn1 = Button(self.page,text='本地单文件识别,识别方式--MFCC',command=self.com1,width = 60)
        btn1.grid(row=3)
        btn2 = Button(self.page,text='本地语音集识别',command=self.com2,width = 60)
        btn2.grid(row=5)
        btn3 = Button(self.page,text='本地单文件识别,识别方式--语谱图',command=self.com3,width = 60)
        btn3.grid(row=4)
    def com1(self):
        #self.page.destroy()
        MainPage(self.root)
        
    def com2(self):
        
        MainPage2(self.root)
        
    def com3(self):
        
        MainPage3(self.root)