import cv2
from PIL import Image, ImageTk
from glob import glob
import os
from haishoku.haishoku import Haishoku
import numpy as np
import math

#超参数
jpg_dir="E:/cs576/project/data/Data_jpg/interview/interview_0/"


#dominant color
rlist=[]
glist=[]
blist=[]

#motion; optical flow

#角点检测参数
feature_params = dict(maxCorners=100, qualityLevel=0.1, minDistance=7, blockSize=7)

#KLT光流参数
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.02))

tracks = []
track_len = 15
detect_interval = 5

beforeGrey=None
MotionList=[]

for i in range(0,480):

    jpg_path=jpg_dir+"frame"+str(i)+".jpg"

    print(jpg_path)

    #dominant color
    d_color=Haishoku.getDominant(jpg_path)
    print(i,":",d_color)
    #Haishoku.showDominant(jpg_path)
    rlist.append(d_color[0])
    glist.append(d_color[1])
    blist.append(d_color[2])

    #motion
    img=cv2.imread(jpg_path)

    curGrey=next = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    img0, img1 = beforeGrey, curGrey
    p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
    if len(tracks) > 0:
        img0, img1 = beforeGrey, curGrey
        p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
        # 上一帧的角点和当前帧的图像作为输入来得到角点在当前帧的位置
        p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)

        sum=0.0

        for i in range(0,len(p0)):
            dist=math.sqrt((p0[i][0][0]-p1[i][0][0])**2+(p0[i][0][1]-p1[i][0][1])**2)
            sum=sum+dist

        avg=sum/len(p0)
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

    # 每隔 detect_interval 时间检测一次特征点
    if i % detect_interval == 0:
        mask = np.zeros_like(curGrey)
        mask[:] = 255

        p = cv2.goodFeaturesToTrack(curGrey, mask=mask, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                tracks.append([(x, y)])

    beforeGrey = curGrey


#得到整个视频的dominant color
rlist.sort()
glist.sort()
blist.sort()

print('r:',rlist[240],'g:',glist[240],'b:',blist[240])

#Motion
#print('MotionList:',MotionList)
MotionList.sort()
MotionScore=MotionList[len(MotionList)// 2]
print('MotionScore:',MotionScore)






