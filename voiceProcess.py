import wave
from pydub import AudioSegment
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import numpy as np


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


wave_date, time = wave_read("E:/cs576/project/data/Data_wav/concerts/concerts_0.wav")
x = date_fft(wave_date, time, 1, 2)

wave_date, time = wave_read("E:/cs576/project/data/Data_wav/concerts/concerts_3.wav")
y = date_fft(wave_date, time, 1, 2)

sum1 = 0
sum2 = 0

for i in range(len(x) - 4021):
    sum1 += x[i]
    sum2 += y[i]

print(sum1 / 19980)
print(sum2 / 19980)

print('avg1:',get_voice_descriptor("E:/cs576/project/data/Data_wav/concerts/concerts_0.wav"))
