import cv2
from PIL import Image, ImageTk
from glob import glob
import os
from haishoku.haishoku import Haishoku
import numpy as np
import math
import matplotlib.pyplot as plt
import json


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


def findTop5(input):

    #超参数
    jpg_root = "E:/cs576/project/data/Data_jpg/"
    voice_root = "E:/cs576/project/data/Data_wav/"
    json_path="../dataBase/data.json"

    input = input

    jpg_dir = jpg_root + input.split("_")[0] + "/" + input + "/"
    voice_dir = voice_root + input.split("_")[0] + input + ".wav"

    Hlist, Vlist, MotionList, dhashList = buildingDatabase(jpg_dir)

    with open(json_path, 'r') as load_f:
        load_dict = json.load(load_f)
        print(load_dict)


    compareDic={}

    for key in load_dict:

        Hlist2=load_dict[key]['Hlist']
        Vlist2 = load_dict[key]['Vlist']
        MotionList2 = load_dict[key]['MotionList']
        dhashList2 = load_dict[key]['dhashList']

        Hsimi=countError(Hlist,Hlist2)
        Vsimi = countError(Vlist, Vlist2)
        MotionSimi = countError(MotionList, MotionList2)
        dhashSimi = countErrorHash(dhashList, dhashList2)

        allSimi=0.4*Hsimi+0.3*MotionSimi+0.1*Vsimi+0.2*dhashSimi

        compareDic[key]={'allSimi':allSimi,'Hsimi':Hsimi,'Vsimi':Vsimi,'MotionSimi':MotionSimi,'dhashSimi':dhashSimi}

    compareList=sorted(compareDic.items(), key=lambda item: item[1]['allSimi'])

    return compareList

input="concerts_0"
compareList=findTop5(input)

print('compareDic:',compareList)

count = 0
for key, value in compareList:
    print(key, ':', value)
    count += 1
    if count == 5:
        break








