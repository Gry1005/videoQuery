import cv2 as cv
import numpy as np

f = open("E:/cs576/project/data/Data_rgb/ads/ads_0/frame0.rgb", "rb")
data = f.read()
f.close()

width=640
height=360

img_out=np.zeros((height,width,3),dtype=np.uint8)

ind = 0
for y in range(0,height):
    for x in range(0,width):
        r = data[ind]
        g = data[ind + height * width]
        b = data[ind + height * width * 2]

        img_out[y][x][0] = b
        img_out[y][x][1] = g
        img_out[y][x][2] = r

        ind+=1


cv.imshow("data", img_out)
cv.waitKey()