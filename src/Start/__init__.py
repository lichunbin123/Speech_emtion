from tkinter import *
from tkinter.messagebox import showinfo, showwarning
import tkinter.filedialog
from PIL import ImageTk
from First_page import *

#初始化
#通过创建对象root来进行后续界面的创建
#设定窗口大小

root = Tk()
root.title('语音情感识别小程序')
root.geometry('520x450')
#加载图片
img = ImageTk.PhotoImage(file='python.jpg')
lb_image = Label(root,image=img)
lb_image.grid(row=0)
#进入第一个页面
First_page(root)
root.mainloop()