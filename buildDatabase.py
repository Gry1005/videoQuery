import cv2
from PIL import Image, ImageTk
from glob import glob
import os
from haishoku.haishoku import Haishoku
import numpy as np
import math
import matplotlib.pyplot as plt
import json
import wave
from pydub import AudioSegment
from scipy.fftpack import fft


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

def buildingDatabase(jpg_dir,wav_path):
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

    #voice
    voiceValue = get_voice_descriptor(wav_path)
    print('voiceValue:',voiceValue)

    return Hlist,Vlist,MotionList,dhashList,voiceValue

#voice processing
def wave_read(path):
    # 打开wav文件 ，open返回一个的是一个Wave_read类的实例，
    # 通过调用它的方法读取WAV文件的格式和数据
    # 截取前20秒音频
    newpath = './test.wav'
    start_time = 0
    end_time = 20000
    sound = AudioSegment.from_wav(path)
    word = sound[start_time:end_time]
    word.export(newpath, format="wav")

    f = wave.open(newpath, "rb")
    # 一次性返回所有的WAV文件的格式信息，它返回的是一个组元(tuple)：声道数, 量化位数（byte单位）,
    # 采样频率, 采样点数, 压缩类型, 压缩类型的描述。wave模块只支持非压缩的数据，因此可以忽略最后两个信息
    params = f.getparams()
    # 读取波形数据
    nchannels, sampwidth, framerate, nframes = params[:4]
    # 读取声音数据，传递一个参数指定需要读取的长度（以取样点为单位）
    str_date = f.readframes(nframes)
    f.close()
    # 需要根据声道数和量化单位，将读取的二进制数据转换为一个可以计算的数组
    wave_date = np.frombuffer(str_date, dtype=np.short)
    # 将wave_data数组改为2列，行数自动匹配。在修改shape的属性时，需使得数组的总长度不变。
    wave_date.shape = -1, 2
    # 转置数据,使成为2行的数据，方便下面时间匹配
    wave_date = wave_date.T
    # 通过取样点数和取样频率计算出每个取样的时间,也就是周期T=采样单数/采样率
    time = np.arange(0, nframes) * (1.0 / framerate)
    return wave_date, time


def date_fft(data, time, start, end):
    t = []
    y = []
    for i in range(time.size):
        if (time[i] >= start) & (time[i] <= end):
            t = np.append(t, time[i])
            y = np.append(y, data[0][i])  # 取左声道
    n = len(t)  # 信号长度
    yy = fft(y)
    yf = abs(yy)  # 取绝对值
    yf1 = abs(fft(y)) / n  # 归一化处理
    yf2 = yf1[range(int(n / 2))]  # 由于对称性，只取一半区间

    xf = np.arange(len(y))  # 频率
    xf1 = xf
    xf2 = xf[range(int(n / 2))]  # 取一半区间

    return yf2

    # 显示原始序列
    plt.figure()
    plt.subplot(221)
    plt.plot(t, y, 'g')
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Original wave")

    # 显示取绝对值后的序列
    plt.subplot(222)
    plt.plot(xf, yf)
    plt.xlabel("Freq (Hz)")
    plt.ylabel("|Y(freq)|")
    plt.title("FFT of Mixed wave(two sides frequency range", fontsize=7, color='#7A378B')
    # 注意这里的颜色可以查询颜色代码表

    # 显示归一化处理后双边序列
    plt.subplot(223)
    plt.plot(xf1, yf1)
    # 注意这里的颜色可以查询颜色代码表
    plt.xlabel("Freq (Hz)")
    plt.ylabel("|Y(freq)|")
    plt.title('FFT of Mixed wave(Normalized processing)', fontsize=10, color='#F08080')

    # 显示归一化处理后单边序列
    plt.subplot(224)
    plt.plot(xf2, yf2, 'b')
    # 注意这里的颜色可以查询颜色代码表
    plt.xlabel("Freq (Hz)")
    plt.ylabel("|Y(freq)|")
    plt.title('FFT of Mixed wave', fontsize=10, color='#F08080')

    plt.show()

def get_voice_descriptor(wav_path):

    wave_date, time = wave_read(wav_path)
    x = date_fft(wave_date, time, 1, 2)

    sum=0

    for i in range(len(x) - 4021):
        sum += x[i]

    return sum/19980


jpg_root="E:/cs576/project/data/Data_jpg/"
voice_root="E:/cs576/project/data/Data_wav/"

cate={'ads':[0,1,2,3],'cartoon':[0,1,2,3,4],'concerts':[0,1,2,3],'interview':[0,1,2,3,4,5],'movies':[0,1,2,3,4],'sport':[0,1,2,4,5]}

jsonDict={}

for key in cate.keys():

    jpg_dir=jpg_root+key+"/"
    voice_dir=voice_root+key+"/"

    iRange=cate[key]

    for i in iRange:

        item=key+"_"+str(i)

        print('item:',item)

        jpg_path = jpg_dir + item + "/"
        wav_path = voice_dir + item + ".wav"

        print(jpg_path)
        print(wav_path)

        Hlist, Vlist, MotionList, dhashList, voiceValue = buildingDatabase(jpg_path)

        # build json
        jsonDict[item]={'jpgPath':jpg_path,'wavPath':wav_path,'Hlist':Hlist,'Vlist':Vlist,'MotionList':MotionList,'dhashList':dhashList,'voiceValue':voiceValue}

        # 作图
        '''
        x = [i for i in range(0, 480)]

        figSize = (10, 5)

        plt.figure(1, figsize=figSize)
        plt.title('H:')
        plt.plot(x, Hlist)
        plt.show()

        plt.figure(2, figsize=figSize)
        plt.title('V:')
        plt.plot(x, Vlist)
        plt.show()

        plt.figure(3, figsize=figSize)
        plt.title('Motion:')
        plt.plot(x, MotionList)
        plt.show()
        '''

#json_str = json.dumps(jsonDict)
#new_dict = json.loads(json_str)

with open("../dataBase/data.json","w") as f:
    json.dump(jsonDict,f)
    print("加载入文件完成...")















