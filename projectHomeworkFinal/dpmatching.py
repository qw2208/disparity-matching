from PIL import Image
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal
from scipy.signal import convolve2d as conv2d
import time
# import cv2

## Use the function below to dynamically change image size.
def create_img(filename,):
        im = Image.open(filename).convert('L') #.convert('L') converts the image to grayscale
        # im.save('AAA.jpg')
        return np.array(im).astype(np.int32)
def convCPU(matrix, kernel):
    convRes = conv2d(matrix,kernel, mode='same')
    return convRes
# mask = np.array([[0., 1., 2. , 1., 0.], [1., 4., 8., 4., 1.],[2.,8.,16.,8.,2.],[1.,4.,8.,4.,1.], [0.,1., 2., 1.,0.]]).astype(np.int32)
mask = np.ones((6,6),dtype=np.int16)
imageL = create_img("imL.png")
#  imageL = convCPU(imageL,mask)
imageR = create_img("imR.png")
# imageR = convCPU(imageR,mask)
height = imageL.shape[0]
width = imageR.shape[1]

start = time.time()
# print imageL.shape
depthMap = []
# print list(imageL[0])
# print list(imageR[0])
for line in xrange(height):

    disparityMap = np.zeros((width,40),dtype=np.int16)

    for i in xrange(width):
        for j in xrange(40): 
            if i+j < 20:
                disparityMap[i][j] = 0
            elif  i+j > width+20:
                disparityMap[i][j] = 10000
            elif i > 0 and j > 0 and j < 39 and i+j < width+20 and i+j > 20:
                disparityMap[i][j] = min(disparityMap[i-1][j],disparityMap[i-1][j+1],disparityMap[i][j-1]) + np.abs(imageL[line][i]-imageR[line][i+j-20])
            elif i > 0 and j == 39 and i+j < width+20 and i+j > 20:
                disparityMap[i][j] = min(disparityMap[i-1][j],disparityMap[i][j-1]) + np.abs(imageL[line][i]-imageR[line][i+j-20])
            elif i > 0 and j == 0 and i+j < width+20 and i+j > 20:
                disparityMap[i][j] = min(disparityMap[i-1][j],disparityMap[i-1][j+1]) + np.abs(imageL[line][i]-imageR[line][i+j-20])
            elif i == 0 and i+j < width+20 and i+j > 20:
                disparityMap[i][j] = disparityMap[i][j-1] + np.abs(imageL[line][i]-imageR[line][i+j-20])
    # print list(disparityMap)

    startX = 20
    startY = width-1

    dispRes = np.ones((width,40))

    while startY > 0:
        # print startY
        if startX < 39 and startX > 0:
            if disparityMap[startY-1][startX] <= min(disparityMap[startY-1][startX+1], disparityMap[startY][startX-1]):
                dispRes[startY][startX] = 0
                startY -= 1
            elif disparityMap[startY-1][startX+1] <= min(disparityMap[startY-1][startX], disparityMap[startY][startX-1]):
                startX += 1
                startY -= 1
            elif disparityMap[startY][startX-1] <= min(disparityMap[startY-1][startX+1], disparityMap[startY-1][startX]):
                startX -= 1
        elif startX == 0:
            if disparityMap[startY-1][startX] <= (disparityMap[startY-1][startX+1]):
                dispRes[startY][startX] = 0
                startY -= 1
            elif disparityMap[startY-1][startX+1] <= (disparityMap[startY-1][startX]):
                startX += 1
                startY -= 1
        elif startX == 39:
            if disparityMap[startY-1][startX] <= disparityMap[startY][startX-1]:
                dispRes[startY][startX] = 0
                startY -= 1
            elif disparityMap[startY][startX-1] <= disparityMap[startY-1][startX]:
                startX -= 1
    # print list(dispRes)

    depthMap1 = []

    for i in xrange(width):
        try:
            col = list(dispRes[i]).index(0)
            depthMap1.append(abs(col-20))
        except:
            depthMap1.append(0)

    depthMap.append(depthMap1)
print "execution time", time.time()-start
depthMap = np.array(depthMap).astype(np.int16)
# print list(depthMap)
# depthMap = convCPU(depthMap,mask)
depthMap = depthMap*255//np.max(depthMap)

new_im = Image.fromarray(depthMap)
new_im.convert('RGB').save('depth.jpg')


















