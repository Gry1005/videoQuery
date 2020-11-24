import pygame as py
import _thread
import time
import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import multiprocessing
from pydub import AudioSegment

#GUI代码

#显示参数
window_width = 960
window_height = 720
image_width = int(window_width * 0.5)
image_height = int(window_height * 0.5)
imagepos_x = 0
imagepos_y = 25

#控制视频的参数
isPause1=True
isPause2=True

index1=0
index2=0

#控制音频的参数
#0 什么都没有播放，1 是正在播放
flag=0
flag2=0

#文件目录
video_dir="E:/cs576/project/data/Data_jpg/"
video1_path=video_dir+"ads/ads_0/"


voice_dir="E:/cs576/project/data/Data_wav/"
voice1_path=voice_dir+"ads/ads_0.wav"


listboxDic={
    'cartoon_0':['E:/cs576/project/data/Data_jpg/cartoon/cartoon_0/',"E:/cs576/project/data/Data_wav/cartoon/cartoon_0.wav"],
    'cartoon_1':['E:/cs576/project/data/Data_jpg/cartoon/cartoon_1/',"E:/cs576/project/data/Data_wav/cartoon/cartoon_1.wav"],
    'cartoon_2':['E:/cs576/project/data/Data_jpg/cartoon/cartoon_2/',"E:/cs576/project/data/Data_wav/cartoon/cartoon_2.wav"],
}

video2_path=listboxDic['cartoon_0'][0]
voice2_path=listboxDic['cartoon_0'][1]

#音频加载
py.mixer.init()
sound1=py.mixer.Sound(voice1_path)
sound2=py.mixer.Sound(voice2_path)

a=py.mixer.get_num_channels()  #获取本机的音频通道数
print('num of channels: ',a)

ch1=py.mixer.Channel(0)   #创建一个Channel对象
ch2=py.mixer.Channel(1)


#图像转换，用于在画布中显示
def tkImage(videoPath,videoInd):

    global index1,index2, isPause1, isPause2, flag,flag2

    if videoInd==1:
        image_path = videoPath + "frame" + str(index1) + ".jpg"
        pilImage = Image.open(image_path)
        pilImage = pilImage.resize((image_width, image_height), Image.ANTIALIAS)
        tkImage = ImageTk.PhotoImage(image=pilImage)

        sca1.set(index1)

        if index1 < 479:
            index1 += 1
        else:
            index1 = 0
            isPause1=True
            ch1.stop()
            flag=0

        return tkImage

    else:
        image_path = videoPath + "frame" + str(index2) + ".jpg"
        pilImage = Image.open(image_path)
        pilImage = pilImage.resize((image_width, image_height), Image.ANTIALIAS)
        tkImage = ImageTk.PhotoImage(image=pilImage)

        sca2.set(index2)

        if index2 < 479:
            index2 += 1
        else:
            index2 = 0
            isPause2=True
            ch2.stop()
            flag2=0

        return tkImage



# 图像的显示与更新
def video():
    def video_loop():
        try:
            while True:
                if not isPause1:
                    # 获取图片
                    picture1 = tkImage(video1_path, 1)
                    canvas1.create_image(0, 0, anchor='nw', image=picture1)

                if not isPause2:
                    picture2 = tkImage(video2_path, 2)
                    canvas2.create_image(0, 0, anchor='nw', image=picture2)

                time.sleep(0.018)

                win.update_idletasks()  # 最重要的更新是靠这两句来实现
                win.update()

        except:
            pass

    video_loop()
    win.mainloop()
    cv2.destroyAllWindows()


win = tk.Tk()
win.geometry(str(window_width) + 'x' + str(window_height))

leftFrm=tk.Frame(win,width=480,height=720)
leftFrm.place(x=0,y=0)
rightFrm=tk.Frame(win,width=480,height=720)
rightFrm.place(x=480,y=0)

#画布
canvas1 = Canvas(leftFrm, bg='white', width=image_width, height=image_height)
canvas1.place(x=imagepos_x, y=imagepos_y+100)

canvas2 = Canvas(rightFrm, bg='white', width=image_width, height=image_height)
canvas2.place(x=imagepos_x, y=imagepos_y+100)

#Label
lb1 = Label(leftFrm,text='query: '+'ads_0')
lb1.place(x=imagepos_x+100,y=imagepos_y)

#Listbox
def lb_click(event):
    global isPause2,index2,flag2,video2_path,voice2_path,sound2

    isPause2=True
    index2=0
    flag2=0
    ch2.stop()

    w = event.widget
    curIndex = w.nearest(event.y)
    selectedKey = w.get(curIndex)

    #print('key:',selectedKey)

    video2_path=listboxDic[selectedKey][0]
    voice2_path=listboxDic[selectedKey][1]
    sound2 = py.mixer.Sound(voice2_path)


var2=tk.StringVar()
var2.set(['cartoon_0','cartoon_1','cartoon_2'])
lb=Listbox(rightFrm,listvariable=var2,height=5)
lb.bind('<Button-1>', lb_click)
lb.place(x=imagepos_x+0,y=imagepos_y)

#Button pause
def p_b1_click():
    global isPause1
    isPause1=True
    ch1.pause()

p_b1=Button(leftFrm,text='Pause',command=p_b1_click)
p_b1.place(x=imagepos_x+100,y=imagepos_y+600)

def p_b2_click():
    global isPause2
    isPause2=True
    ch2.pause()

p_b2=Button(rightFrm,text='Pause',command=p_b2_click)
p_b2.place(x=imagepos_x+100,y=imagepos_y+600)

#Button start
def start_b1_click():
    global isPause1,flag
    isPause1=False
    if flag==0:
        ch1.play(sound1,loops=-1)
        flag=1
    else:
        ch1.unpause()

start_b1=Button(leftFrm,text='Start',command=start_b1_click)
start_b1.place(x=imagepos_x+0,y=imagepos_y+600)

def start_b2_click():
    global isPause2,flag2
    isPause2=False
    if flag2==0:
        ch2.play(sound2,loops=-1)
        flag2=1
    else:
        ch2.unpause()

start_b2=Button(rightFrm,text='Start',command=start_b2_click)
start_b2.place(x=imagepos_x+0,y=imagepos_y+600)

#Button stop
def stop_b1_click():
    global isPause1,flag,index1
    isPause1=True
    index1=0
    flag=0
    ch1.stop()

stop_b1=Button(leftFrm,text='Stop',command=stop_b1_click)
stop_b1.place(x=imagepos_x+200,y=imagepos_y+600)

def stop_b2_click():
    global isPause2,flag2,index2
    isPause2=True
    index2=0
    flag2=0
    ch2.stop()

stop_b2=Button(rightFrm,text='Stop',command=stop_b2_click)
stop_b2.place(x=imagepos_x+200,y=imagepos_y+600)

#Scale
sca1=tk.Scale(leftFrm,from_=0,to_=480,orient=tk.HORIZONTAL,length=480,showvalue=True,tickinterval=120,resolution=1)
sca1.place(x=imagepos_x+0,y=imagepos_y+520)

sca2=tk.Scale(rightFrm,from_=0,to_=480,orient=tk.HORIZONTAL,length=480,showvalue=True,tickinterval=120,resolution=1)
sca2.place(x=imagepos_x+0,y=imagepos_y+520)

if __name__ == '__main__':

    #用一个函数，得到listboxDic

    p1 = multiprocessing.Process(target=video)
    p1.start()





