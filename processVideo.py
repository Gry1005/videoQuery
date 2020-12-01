import pygame as py
import _thread
import time
import tkinter as tk
from tkinter import *
import cv2
import multiprocessing
from pydub import AudioSegment
import os
from PIL import Image, ImageTk
from haishoku.haishoku import Haishoku
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import json

#GUI代码

#显示参数
window_width = 1500
window_height = 800
#640,360
image_width = int(640*0.5)
image_height = int(360*0.5)
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
video1_files=None
video2_files=None

video_dir="E:/cs576/project/data/Data_jpg/"
#video1_path=video_dir+"ads/ads_0/"
video1_path=""
input=""

voice_dir="E:/cs576/project/data/Data_wav/"
voice1_path=None

'''
listboxDic={
    'cartoon_0':['E:/cs576/project/data/Data_jpg/cartoon/cartoon_0/',"E:/cs576/project/data/Data_wav/cartoon/cartoon_0.wav"],
    'cartoon_1':['E:/cs576/project/data/Data_jpg/cartoon/cartoon_1/',"E:/cs576/project/data/Data_wav/cartoon/cartoon_1.wav"],
    'cartoon_2':['E:/cs576/project/data/Data_jpg/cartoon/cartoon_2/',"E:/cs576/project/data/Data_wav/cartoon/cartoon_2.wav"],
    'sport_5':['E:/cs576/project/data/Data_jpg/sport/sport_5/','E:/cs576/project/data/Data_wav/sport/sport_5.wav']
}

video2_path=listboxDic[list(listboxDic.keys())[0]][0]
voice2_path=listboxDic[list(listboxDic.keys())[0]][1]
'''
listboxDic={}
video2_path=None
voice2_path=None

#存储数据记录
Hlist=None
Vlist=None
MotionList=None
video1_dhash_list=None

Hlist2=None
Vlist2=None
MotionList2=None
video2_dhash_list=None

#音频加载
py.mixer.init()
'''
sound1=py.mixer.Sound(voice1_path)
sound2=py.mixer.Sound(voice2_path)
'''
sound1=None
sound2=None

a=py.mixer.get_num_channels()  #获取本机的音频通道数
print('num of channels: ',a)

ch1=py.mixer.Channel(0)   #创建一个Channel对象
ch2=py.mixer.Channel(1)

def tkImage(video_path):

    image_path = video_path

    pilImage = Image.open(image_path)
    pilImage = pilImage.resize((image_width, image_height), Image.ANTIALIAS)
    tkImage = ImageTk.PhotoImage(image=pilImage)

    return tkImage


# 图像的显示与更新
def video(input):
    global var2, video1_path, video2_path, index1, index2, flag, flag2, isPause1, isPause2, sound1,sound2, video1_files, video2_files,voice1_path, voice2_path

    print('processing...')

    lb1.config(text=input)

    video1_path = video_dir + input.split("_")[0] + "/" + input + "/"
    #print('video1_path:', video1_path)

    # 用一个函数，得到listboxDic
    videoProcessing(video1_path)

    #print(listboxDic)

    video2_path = listboxDic[list(listboxDic.keys())[0]]['jpgPath']
    voice2_path = listboxDic[list(listboxDic.keys())[0]]['wavPath']

    var2.set(list(listboxDic.keys()))


    video1_files = os.listdir(video1_path)
    video1_files.sort(key=lambda x: int(x.split('.')[0][5:]))
    # print(video1_files)

    voice1_path = voice_dir + input.split("_")[0] + "/" + input + ".wav"
    sound1 = py.mixer.Sound(voice1_path)
    #print('voice1_path:', voice1_path)

    video2_files = os.listdir(video2_path)
    video2_files.sort(key=lambda x: int(x.split('.')[0][5:]))

    sound2 = py.mixer.Sound(voice2_path)

    # print(video2_files)

    def video_loop():
        global video1_path, video2_path, index1, index2, flag, flag2, isPause1, isPause2, sound1, video1_files, video2_files

        try:
            while True:
                if not isPause1:
                    # 获取图片
                    picture1 = tkImage(video1_path+video1_files[index1])
                    canvas1.create_image(0, 0, anchor='nw', image=picture1)

                    sca1.set(index1)
                    lb_dhash_left.config(text="dhash: "+video1_dhash_list[index1])
                    lb_canvs_1.config(text="H: "+str(Hlist[index1])+" V:"+str(Vlist[index1])+" Motion: "+str(MotionList[index1]))

                    if index1 < 479:
                        index1 += 1
                    else:
                        index1 = 0
                        isPause1 = True
                        ch1.stop()
                        flag = 0

                if not isPause2:
                    picture2 = tkImage(video2_path+video2_files[index2])
                    canvas2.create_image(0, 0, anchor='nw', image=picture2)

                    sca2.set(index2)
                    lb_dhash_right.config(text="dhash: " + video2_dhash_list[index2])
                    lb_canvs_2.config(text="H: " + str(Hlist2[index2]) + " V:" + str(Vlist2[index2]) + " Motion: " + str(MotionList2[index2]))

                    if index2 < 479:
                        index2 += 1
                    else:
                        index2 = 0
                        isPause2 = True
                        ch2.stop()
                        flag2 = 0


                time.sleep(0.015)

                win.update_idletasks()  # 最重要的更新是靠这两句来实现
                win.update()

        except:
            pass

    video_loop()
    win.mainloop()
    cv2.destroyAllWindows()


win = tk.Tk()
win.geometry(str(window_width) + 'x' + str(window_height))

leftFrm=tk.Frame(win,width=window_width//2,height=window_height)
leftFrm.place(x=0,y=0)
rightFrm=tk.Frame(win,width=window_width//2,height=window_height)
rightFrm.place(x=window_width//2,y=0)

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
    global isPause2,index2,flag2,video2_path,voice2_path,sound2,video2_files,canvs4,canvs5,canvs6,Hlist2,Vlist2,MotionList2,video2_dhash_list

    isPause2=True
    index2=0
    flag2=0
    ch2.stop()

    w = event.widget
    curIndex = w.nearest(event.y)
    selectedKey = w.get(curIndex)

    #print('key:',selectedKey)

    video2_path=listboxDic[selectedKey]['jpgPath']
    voice2_path=listboxDic[selectedKey]['wavPath']

    video2_files = os.listdir(video2_path)
    #print('video2_path:',video2_path)
    video2_files.sort(key=lambda x: int(x.split('.')[0][5:]))
    #print('video2_files:', video2_files)

    sound2 = py.mixer.Sound(voice2_path)

    # 作图
    Hlist2 = listboxDic[selectedKey]['Hlist']
    Vlist2 = listboxDic[selectedKey]['Vlist']
    MotionList2 = listboxDic[selectedKey]['MotionList']
    video2_dhash_list = listboxDic[selectedKey]['dhashList']

    x = [i for i in range(0, 480)]

    canvs4.get_tk_widget().destroy()
    canvs5.get_tk_widget().destroy()
    canvs6.get_tk_widget().destroy()

    f4 = Figure(figsize=figSize, dpi=dpi)
    f4_plot = f4.add_subplot(111)
    f4_plot.xaxis.set_visible(False)
    f4_plot.plot(x, Hlist2)
    # canvs = FigureCanvasTkAgg(f1, leftFrm)
    canvs4 = FigureCanvasTkAgg(f4, rightFrm)
    canvs4.get_tk_widget().place(x=canvs_x, y=canvs_y)

    f5 = Figure(figsize=figSize, dpi=dpi)
    f5_plot = f5.add_subplot(111)
    f5_plot.xaxis.set_visible(False)
    f5_plot.plot(x, Vlist2)
    canvs5 = FigureCanvasTkAgg(f5, rightFrm)
    canvs5.get_tk_widget().place(x=canvs_x, y=canvs_y+100)

    f6 = Figure(figsize=figSize, dpi=dpi)
    f6_plot = f6.add_subplot(111)
    f6_plot.xaxis.set_visible(False)
    f6_plot.plot(x, MotionList2)
    canvs6 = FigureCanvasTkAgg(f6, rightFrm)
    canvs6.get_tk_widget().place(x=canvs_x, y=canvs_y+200)


var2=tk.StringVar()
#var2.set(list(listboxDic.keys()))
lb=Listbox(rightFrm,listvariable=var2,height=5)
lb.bind('<Button-1>', lb_click)
lb.place(x=imagepos_x+0,y=imagepos_y)

#Button pause
def p_b1_click():
    global isPause1
    isPause1=True
    ch1.pause()

p_b1=Button(leftFrm,text='Pause',command=p_b1_click)
p_b1.place(x=imagepos_x+100,y=imagepos_y+700)

def p_b2_click():
    global isPause2
    isPause2=True
    ch2.pause()

p_b2=Button(rightFrm,text='Pause',command=p_b2_click)
p_b2.place(x=imagepos_x+100,y=imagepos_y+700)

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
start_b1.place(x=imagepos_x+0,y=imagepos_y+700)

def start_b2_click():
    global isPause2,flag2
    isPause2=False
    if flag2==0:
        ch2.play(sound2,loops=-1)
        flag2=1
    else:
        ch2.unpause()

start_b2=Button(rightFrm,text='Start',command=start_b2_click)
start_b2.place(x=imagepos_x+0,y=imagepos_y+700)

#Button stop
def stop_b1_click():
    global isPause1,flag,index1
    isPause1=True
    index1=0
    flag=0
    ch1.stop()

stop_b1=Button(leftFrm,text='Stop',command=stop_b1_click)
stop_b1.place(x=imagepos_x+200,y=imagepos_y+700)

def stop_b2_click():
    global isPause2,flag2,index2
    isPause2=True
    index2=0
    flag2=0
    ch2.stop()

stop_b2=Button(rightFrm,text='Stop',command=stop_b2_click)
stop_b2.place(x=imagepos_x+200,y=imagepos_y+700)

#Scale
sca1=tk.Scale(leftFrm,from_=0,to_=480,orient=tk.HORIZONTAL,length=600,showvalue=True,tickinterval=120,resolution=1)
sca1.place(x=imagepos_x+110,y=imagepos_y+630)

sca2=tk.Scale(rightFrm,from_=0,to_=480,orient=tk.HORIZONTAL,length=600,showvalue=True,tickinterval=120,resolution=1)
sca2.place(x=imagepos_x+110,y=imagepos_y+630)

#折线图
figSize = (8, 1)
dpi = 100
canvs_x=0
canvs_y=imagepos_y + 325

canvs=None
canvs2=None
canvs3=None
canvs4=None
canvs5=None
canvs6=None

lb_canvs_1=tk.Label(leftFrm,text="H: V: Motion:")
lb_canvs_1.place(x=0,y=imagepos_y+285)

lb_canvs_2=tk.Label(rightFrm,text="H: V: Motion:")
lb_canvs_2.place(x=0,y=imagepos_y+285)



#dhashLabel
lb_dhash_left=tk.Label(leftFrm,text="dhash:")
lb_dhash_left.place(x=0,y=imagepos_y+305)

lb_dhash_right=tk.Label(rightFrm,text="dhash:")
lb_dhash_right.place(x=0,y=imagepos_y+305)

#处理视频

# 超参数
json_path = "../dataBase/data.json"
load_dict=None

with open(json_path, 'r') as load_f:
    load_dict = json.load(load_f)
    #print(load_dict)

# 差异值哈希算法
def d_hash(img):
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    s = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                s += '1'
            else:
                s += '0'
    result = ''
    for i in range(0, 64, 4):
        result += ''.join('%x' % int(s[i: i + 4], 2))
    return result


# 计算汉明距离
def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        return
    count = 0
    for i in range(len(str1)):
        if str1[i] != str2[i]:
            count += 1
    return count


def buildingDatabase(jpg_dir):
    # 超参数
    jpg_dir = jpg_dir
    image_files = os.listdir(jpg_dir)
    image_files.sort(key=lambda x: int(x.split('.')[0][5:]))

    # 纹理检测

    dhashList = []

    # dominant color
    Hlist = []
    # Slist=[]
    Vlist = []

    # motion; optical flow

    # 角点检测参数
    feature_params = dict(maxCorners=100, qualityLevel=0.1, minDistance=7, blockSize=7)

    # KLT光流参数
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.02))

    tracks = []
    track_len = 15
    detect_interval = 5

    beforeGrey = None
    MotionList = []

    count = 0
    # for i in range(0,480):
    for jpg_path in image_files[0:480]:

        # jpg_path=jpg_dir+"frame"+str(i)+".jpg"
        jpg_path = jpg_dir + jpg_path

        print(jpg_path)

        # dominant color
        d_color = Haishoku.getDominant(jpg_path)
        #print(count, ':', d_color)

        r = d_color[0]
        g = d_color[1]
        b = d_color[2]

        # 转换为HSV空间
        cmax = max(max(r, g), b)
        cmin = min(min(r, g), b)
        delta = cmax - cmin

        V = cmax
        if cmax == 0:
            S = 0
        else:
            S = delta / cmax

        if delta==0:
            H=0
        elif cmax == r:
            H = ((g - b) / delta) * 60
        elif cmax == g:
            H = 120 + ((b - r) / delta) * 60
        else:
            H = 240 + ((r - g) / delta) * 60

        if H<0:
            H=H+360

        Hlist.append(H)
        # Slist.append(S)
        Vlist.append(V)

        img = cv2.imread(jpg_path)
        # 纹理特征
        dhash = d_hash(img)
        dhashList.append(dhash)

        # motion
        curGrey = next = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img0, img1 = beforeGrey, curGrey
        p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
        if len(tracks) > 0:
            img0, img1 = beforeGrey, curGrey
            p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
            # 上一帧的角点和当前帧的图像作为输入来得到角点在当前帧的位置
            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)

            sum = 0.0

            for i in range(0, len(p0)):
                dist = math.sqrt((p0[i][0][0] - p1[i][0][0]) ** 2 + (p0[i][0][1] - p1[i][0][1]) ** 2)
                sum = sum + dist

            avg = sum / len(p0)
            MotionList.append(avg)

            # 反向检查,当前帧跟踪到的角点及图像和前一帧的图像作为输入来找到前一帧的角点位置
            p0r, _, _ = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

            # 得到角点回溯与前一帧实际角点的位置变化关系
            d = abs(p0 - p0r).reshape(-1, 2).max(-1)

            # 判断d内的值是否小于1，大于1跟踪被认为是错误的跟踪点
            good = d < 1

            new_tracks = []

            for i, (tr, (x, y), flag) in enumerate(zip(tracks, p1.reshape(-1, 2), good)):

                # 判断是否为正确的跟踪点
                if not flag:
                    continue

                # 存储动态的角点
                tr.append((x, y))

                # 只保留track_len长度的数据，消除掉前面的超出的轨迹
                if len(tr) > track_len:
                    del tr[0]
                # 保存在新的list中
                new_tracks.append(tr)

            # 更新特征点
            tracks = new_tracks

        else:
            MotionList.append(0.0)

        # 每隔 detect_interval 时间检测一次特征点
        if count % detect_interval == 0:
            mask = np.zeros_like(curGrey)
            mask[:] = 255

            p = cv2.goodFeaturesToTrack(curGrey, mask=mask, **feature_params)
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    tracks.append([(x, y)])

        beforeGrey = curGrey
        count += 1

    # 得到整个视频的dominant color
    print('H len:', len(Hlist))
    print('Hlist:', Hlist)
    # print('S len:',len(Slist))
    # print('Slist:',Slist)
    print('V len:', len(Vlist))
    print('Vlist:', Vlist)

    # 纹理
    print('hash len:', len(dhashList))
    print('hash list:', dhashList)

    # Motion
    print('Motion len:', len(MotionList))
    print('MotionList:', MotionList)

    return Hlist,Vlist,MotionList,dhashList

def countError(list1,list2):

    list1.sort()
    list2.sort()

    sumSimilar=0

    for i in range(0,480):

        x=float(list1[i])
        y=float(list2[i])

        if max(x,y)==0:
            similar=0
        else:
            similar=(max(x,y)-min(x,y))/max(x,y)

        sumSimilar=sumSimilar+similar

    avgSimi=sumSimilar/480

    return avgSimi

def countErrorHash(list1,list2):

    list1.sort()
    list2.sort()

    sumSimilar = 0

    for i in range(0, 480):

        sumSimilar = sumSimilar + hamming_distance(list1[i],list2[i])/16

    avgSimi = sumSimilar / 480

    return avgSimi

def findTop5(Hlist, Vlist, MotionList, dhashList):

    compareDic={}

    for key in load_dict:

        Hlist2=load_dict[key]['Hlist']
        Vlist2 = load_dict[key]['Vlist']
        MotionList2 = load_dict[key]['MotionList']
        dhashList2 = load_dict[key]['dhashList']

        Hsimi=countError(Hlist.copy(),Hlist2.copy())
        Vsimi = countError(Vlist.copy(), Vlist2.copy())
        MotionSimi = countError(MotionList.copy(), MotionList2.copy())
        dhashSimi = countErrorHash(dhashList.copy(), dhashList2.copy())

        allSimi=0.4*Hsimi+0.3*MotionSimi+0.1*Vsimi+0.2*dhashSimi

        compareDic[key]={'allSimi':allSimi,'Hsimi':Hsimi,'Vsimi':Vsimi,'MotionSimi':MotionSimi,'dhashSimi':dhashSimi}

    compareList=sorted(compareDic.items(), key=lambda item: item[1]['allSimi'])

    return compareList


#视频处理
def videoProcessing(video_path):
    global listboxDic,canvs,canvs2,canvs3,canvs4,canvs5,canvs6,video1_dhash_list,video2_dhash_list,Hlist,Vlist,MotionList,Hlist2,Vlist2,MotionList2

    # 超参数
    jpg_dir = video_path

    Hlist, Vlist, MotionList, dhashList=buildingDatabase(jpg_dir)

    #findTop5
    compareList=findTop5(Hlist, Vlist, MotionList, dhashList)

    #生成listboxDic
    count = 0
    for key, value in compareList:
        print(key, ':', value)

        listboxDic[key]=load_dict[key]

        count += 1
        if count == 5:
            break

    Hlist2=listboxDic[list(listboxDic.keys())[0]]['Hlist']
    Vlist2 = listboxDic[list(listboxDic.keys())[0]]['Vlist']
    MotionList2 = listboxDic[list(listboxDic.keys())[0]]['MotionList']
    dhashList2 = listboxDic[list(listboxDic.keys())[0]]['dhashList']

    video1_dhash_list = dhashList
    video2_dhash_list = dhashList2

    # 作图
    x = [i for i in range(0, 480)]

    f1 = Figure(figsize=figSize, dpi=dpi)
    f1_plot = f1.add_subplot(111)
    f1_plot.xaxis.set_visible(False)
    f1_plot.plot(x, Hlist)
    #canvs = FigureCanvasTkAgg(f1, leftFrm)
    canvs = FigureCanvasTkAgg(f1,leftFrm)
    canvs.get_tk_widget().place(x=canvs_x, y=canvs_y)

    f2 = Figure(figsize=figSize, dpi=dpi)
    f2_plot = f2.add_subplot(111)
    f2_plot.xaxis.set_visible(False)
    f2_plot.plot(x, Vlist)
    canvs2 = FigureCanvasTkAgg(f2, leftFrm)
    canvs2.get_tk_widget().place(x=canvs_x, y=canvs_y+100)

    f3 = Figure(figsize=figSize, dpi=dpi)
    f3_plot = f3.add_subplot(111)
    f3_plot.xaxis.set_visible(False)
    f3_plot.plot(x, MotionList)
    canvs3 = FigureCanvasTkAgg(f3, leftFrm)
    canvs3.get_tk_widget().place(x=canvs_x, y=canvs_y+200)

    f4 = Figure(figsize=figSize, dpi=dpi)
    f4_plot = f4.add_subplot(111)
    f4_plot.xaxis.set_visible(False)
    f4_plot.plot(x, Hlist2)
    # canvs = FigureCanvasTkAgg(f1, leftFrm)
    canvs4 = FigureCanvasTkAgg(f4, rightFrm)
    canvs4.get_tk_widget().place(x=canvs_x, y=canvs_y)

    f5 = Figure(figsize=figSize, dpi=dpi)
    f5_plot = f5.add_subplot(111)
    f5_plot.xaxis.set_visible(False)
    f5_plot.plot(x, Vlist2)
    canvs5 = FigureCanvasTkAgg(f5, rightFrm)
    canvs5.get_tk_widget().place(x=canvs_x, y=canvs_y+100)

    f6 = Figure(figsize=figSize, dpi=dpi)
    f6_plot = f6.add_subplot(111)
    f6_plot.xaxis.set_visible(False)
    f6_plot.plot(x, MotionList2)
    canvs6 = FigureCanvasTkAgg(f6, rightFrm)
    canvs6.get_tk_widget().place(x=canvs_x, y=canvs_y+200)


if __name__ == '__main__':

    input="ads_2"

    p1 = multiprocessing.Process(target=video,kwargs={'input':input})
    p1.start()





