from PIL import Image
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal
from scipy.signal import convolve2d as conv2d
import cv2
from cv2 import *
import time
import numpy as np
from pycuda import driver, compiler, gpuarray, tools

import pycuda.autoinit

#define a function to convert image to array
def create_img(filename):
    im = Image.open(filename).convert('L')
    return np.array(im).astype(np.int32)

#mask= np.ones((3,3)).astype(np.int32)


#print height,width

#get image size here
#print imageL.shape
kernel_code_disparity="""
#include <math.h>
#include <stdlib.h>
#ifndef max
#define max(a,b) (((a)>(b))?(a):(b))
#endif

#ifndef min
#define min(a,b) (((a)<(b))?(a):(b))
#endif
__global__ void inorder(int* imageL, int* imageR, int* depthMap)
{
    const unsigned int height=%(height)s;
    const unsigned int width=%(width)s;
    //int tx=blockIdx.x*blockDim.x+threadIdx.x;
    int tx=blockIdx.x*blockDim.x+threadIdx.x;
    int i;
    int j;
    //__shared__ int disparityMap[width][40];
    //Here is the matrix to store loss
    int disparityMap[width][40];
    //Here is the matrix to store index about disparity
    int disMap[width][40];

    //Calculate the loss matrix
    for(i=0;i<width;i++)
    {
        for(j=0;j<40;j++)
        {
            //pad the upper-left corner
            if(i+j<20)
                disparityMap[i][j]=0;
            //int value=imageL[tx*width+i]-imageR[tx*width+i+j-20];
            //else if(i+j>width+20)
            //    disparityMap[i][j]=1000;

            //middle
            if((i>0)&&(j>0)&&(j<39)&&(i+j<width+20)&&(i+j>=20))
            {
                int value=imageL[tx*width+i]-imageR[tx*width+i+j-20];
                disparityMap[i][j]=min(min(disparityMap[i-1][j],disparityMap[i-1][j+1]),disparityMap[i][j-1])+max(value,-value);
            }
            //right margin
            else if((i>0)&&(j==39)&&(i+j<width+20)&&(i+j>=20))
            {
                int value=imageL[tx*width+i]-imageR[tx*width+i+j-20];
                disparityMap[i][j]=min(disparityMap[i-1][j],disparityMap[i][j-1])+max(value,-value);
            }
            //left margin
            else if((i>0)&&(j==0)&&(i+j<width+20)&&(i+j>=20))
            {
                int value=imageL[tx*width+i]-imageR[tx*width+i+j-20];
                disparityMap[i][j]=min(disparityMap[i-1][j],disparityMap[i-1][j+1])+max(value,-value);
            }
            //up margin
            else if((i==0)&&(i+j<width+20)&&(i+j>=20))
            {
                int value=imageL[tx*width+i]-imageR[tx*width+i+j-20];
                disparityMap[i][j]=disparityMap[i][j-1]+max(value,-value);
            }
        }
    }
    __syncthreads();
    unsigned int startX=20;
    unsigned int startY=width-1;
    //initialize the path matrix
    for(i=0;i<width;i++)
    {
        for(j=0;j<40;j++)
        {
            disMap[i][j]=1;
        }
    }
    __syncthreads();
    //Backwards, record "best" path (mark the matched point with 0)
    while((startY>0))
    {
        //middle
        if((startX < 39)&&(startX>0))
        {
            if(disparityMap[startY-1][startX]<=min(disparityMap[startY-1][startX+1], disparityMap[startY][startX-1]))
            {
                disMap[startY][startX]=0;
                startY-=1;
            }
            else if (disparityMap[startY-1][startX+1] <= min(disparityMap[startY-1][startX], disparityMap[startY][startX-1]))
            {
                startX+=1;
                startY-=1;
            }
            else if (disparityMap[startY][startX-1] <= min(disparityMap[startY-1][startX+1], disparityMap[startY-1][startX]))
                startX-=1;
        }
        //up margin
        else if (startX==0)
        {
            if (disparityMap[startY-1][startX] <= disparityMap[startY-1][startX+1])
            {
                disMap[startY][startX] = 0;
                startY -= 1;
            }
            else if (disparityMap[startY-1][startX+1] <= disparityMap[startY-1][startX])
            {
                startX += 1;
                startY -= 1;
            }
        }
        //down margin
        else if (startX == 39)
        {
            if (disparityMap[startY-1][startX] <= disparityMap[startY][startX-1])
            {
                disMap[startY][startX] = 0;
                startY -= 1;
            }
            else if (disparityMap[startY][startX-1] <= disparityMap[startY-1][startX])
                startX -= 1;
        }
    }
    __syncthreads();
    //Calculate disparity and copy it to global memory
    for(i=0;i<width;i++)
    {
        //int flag = 0;
        for(j=0;j<40;j++)
        {
            if(disMap[i][j]==0)
            {
                depthMap[tx*width+i]=max(20-j,j-20);
        //        flag = 1;
                break;
            }
        }
        //if (flag==0){
        //    if (i==0)
        //        depthMap[tx*width+i]=0;
        //    else
        //        depthMap[tx*width+i]=depthMap[tx*width+i-1];
        //}
    }
}
"""
def DP():
    #load left and right image
    imageL = create_img("imL.png")
    imageR = create_img("imR.png")

    #Get the input size
    height = int(imageL.shape[0])
    width=int(imageL.shape[1])

    #define input in CPU
    imageL_cpu=np.asarray(imageL,dtype=np.int32)
    imageR_cpu=np.asarray(imageR,dtype=np.int32)

    # Transfer host memory to device memory
    imageL_gpu=gpuarray.to_gpu(imageL_cpu)
    imageR_gpu=gpuarray.to_gpu(imageR_cpu)

    #define initial output (GPU)
    depthMap_gpu=gpuarray.zeros((height,width),dtype=np.int32)

    kernel_code=kernel_code_disparity % {
        'height':height,
        'width':width
    }

    mod=compiler.SourceModule(kernel_code)

    disparity=mod.get_function("inorder")

    disparity(
    imageL_gpu,
    imageR_gpu,
    depthMap_gpu,
    grid=(1,1),
    block=(height,1,1),
    )

    depthMap=depthMap_gpu.get()
    # depthMap = np.array(depth).astype(np.int32)

    #Convert range to (0,255)
    depthMap = depthMap*255//20
    # depthMap = cv2.medianBlur(depthMap,7)
    # Use Median Filter to reduce impulse noise
    new_im = Image.fromarray(depthMap)
    new_im.convert('RGB').save('depth.jpg')
    source = cv2.imread("depth.jpg", CV_LOAD_IMAGE_GRAYSCALE)
    final = cv2.medianBlur(source, 3)
    final = Image.fromarray(final)
    final.convert('RGB').save('depth.jpg')

#Calculate Running Timt
M=5
times=[]
for i in xrange(M):
    start = time.time()
    DP()
    times.append(time.time()-start)
dp_time=np.average(times)
print 'DP execution time', dp_time
# disparity(
# imageL_gpu,
# imageR_gpu,
# depthMap_gpu,
# grid=(1,width,1),
# block=(height,1,1),
# )
