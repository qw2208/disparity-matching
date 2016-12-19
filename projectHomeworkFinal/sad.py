# -*- coding: utf-8 -*-
# This is the PyCuda version of tile convolution

from PIL import Image
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import scipy as sp
import scipy.signal
from scipy.signal import convolve2d as conv2d
import cv2
from cv2 import *

from pycuda import driver, compiler, gpuarray, tools
# initialize the device
import pycuda.autoinit
import time


kernel_template = """
// Define constant values
# define HEIGHT %(HEIGHT)s
# define WIDTH %(WIDTH)s
# define KERNEL_SIZE %(KERNEL_SIZE)s
# define TILE_WIDTH %(TILE_WIDTH)s
# define DisparityLevel %(DisparityLevel)s
# define TILE_X %(TILE_X)s
# define TILE_Y %(TILE_Y)s
__global__ void match(int *left, int *right, int *map){
    //Use shared memory to store elements from right image in a block
    __shared__ int right_shared[TILE_Y][TILE_X];
    //Use private memroy to store elements from left image in a window
    int left_private[KERNEL_SIZE][KERNEL_SIZE];
    int i;
    int j;

    //Get the id in the block
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    //Get the id in the whole image
    int row_L=blockIdx.y*TILE_WIDTH+ty;
    int col_L=blockIdx.x*TILE_WIDTH+tx;

    //Get the position in the right image
    int col_R=col_L-DisparityLevel-(KERNEL_SIZE-1)/2;

    //if the element is in the right image
    if ((row_L-(KERNEL_SIZE-1)/2>=0)&&(row_L-(KERNEL_SIZE-1)/2<=HEIGHT)&&(col_R>=0)&&(col_R<=WIDTH))
        right_shared[ty][tx] = right[(row_L-(KERNEL_SIZE-1)/2)*WIDTH+col_R];
    else
        right_shared[ty][tx] = 0;

    __syncthreads();
    //Get the upper-left element(of the window)'s index in left image
    int pos_x=col_L-(KERNEL_SIZE-1)/2;
    int pos_y=row_L-(KERNEL_SIZE-1)/2;
    if ((tx>=0) &&(tx<TILE_WIDTH)&&(ty>=0)&&(ty<TILE_WIDTH)){
        for(i=0;i<KERNEL_SIZE;i++)
        {
            for(j=0;j<KERNEL_SIZE;j++)
            {
                if((pos_y+i<HEIGHT)&&(pos_y+i>=0)&&(pos_x+j>=0)&&(pos_x+j<WIDTH))
                    left_private[i][j] = left[(pos_y+i)*WIDTH+pos_x+j];
                else
                    left_private[i][j] = 0;
            }
        }
        //set initial value of minimum sum of absolute difference
        int minDiff = 65536*KERNEL_SIZE*KERNEL_SIZE;
        //set initial value of minimum offset
        int minOffset = 0;
        for (int offset=0; offset<=DisparityLevel; offset++){
            int currDiff = 0;
            //calculate sum of absolute difference
            for (i=-(KERNEL_SIZE-1)/2; i<=(KERNEL_SIZE-1)/2; i++){
                for (j=-(KERNEL_SIZE-1)/2; j<=(KERNEL_SIZE-1)/2; j++){
                    currDiff += abs(left_private[i+(KERNEL_SIZE-1)/2][j+(KERNEL_SIZE-1)/2]-right_shared[(ty+(KERNEL_SIZE-1)/2)+i][(tx+DisparityLevel+(KERNEL_SIZE-1)/2)+j-offset]);
                    //currDiff += (left_private[i+(KERNEL_SIZE-1)/2][j+(KERNEL_SIZE-1)/2]-right_shared[(ty+(KERNEL_SIZE-1)/2)+i][(tx+DisparityLevel+(KERNEL_SIZE-1)/2)+j-offset])*(left_private[i+(KERNEL_SIZE-1)/2][j+(KERNEL_SIZE-1)/2]-right_shared[(ty+(KERNEL_SIZE-1)/2)+i][(tx+DisparityLevel+(KERNEL_SIZE-1)/2)+j-offset]);
                }
            }
            //update the offset and set the new minimum difference value
            if (currDiff < minDiff){
                minDiff = currDiff;
                minOffset = offset;
            }
        }
        //Copy the result to global memory
        map[(row_L+(KERNEL_SIZE-1)/2)*WIDTH+col_L+(KERNEL_SIZE-1)/2] = minOffset;
    }
}
"""



def imageConv(filenameLeft,filenameRight):
    kernel = np.ones((7,7),dtype=np.int32)
    # kernel = np.random.randint(3,3).astype(np.int32)


    TILE_WIDTH=16   #LEFT IMAGE TILE SIZE
    DisparityLevel=20 #MAX DisparityLevel
    KERNEL_SIZE = kernel.shape[0]   #WINDOW SIZE
    TILE_X=TILE_WIDTH+KERNEL_SIZE+DisparityLevel-1  #RIGHT IMAGE TILE WIDTH
    TILE_Y=TILE_WIDTH+KERNEL_SIZE-1 #RIGHT IMAGE TILE HEIGHT

    # Convert image to grayscale matrix
    matrixLeft = create_img(filenameLeft).astype(np.int32)
    matrixRight = create_img(filenameRight).astype(np.int32)
    HEIGHT = matrixLeft.shape[0]
    WIDTH = matrixLeft.shape[1]


    # Transfer host memory to device memory
    matrix_gpu_left = gpuarray.to_gpu(matrixLeft)
    matrix_gpu_right = gpuarray.to_gpu(matrixRight)
    kernel_gpu = gpuarray.to_gpu(kernel)

    # output_gpu_left = gpuarray.zeros((HEIGHT, WIDTH), np.int32)
    # output_gpu_right = gpuarray.zeros((HEIGHT, WIDTH), np.int32)
    output_gpu_disp = gpuarray.zeros((HEIGHT, WIDTH), np.int32)


    kernel_code = kernel_template % {
    'KERNEL_SIZE':KERNEL_SIZE,
    'HEIGHT':HEIGHT,
    'WIDTH':WIDTH,
    'TILE_WIDTH':TILE_WIDTH,
    'DisparityLevel':DisparityLevel,
    'TILE_X':TILE_X,
    'TILE_Y':TILE_Y,
    }

    # Compile the kernel code
    mod_conv = compiler.SourceModule(kernel_code)
    # Get the kernel function from the compile module
    match = mod_conv.get_function("match")

    #Calculate Kernel part Running Time
    M=5
    times=[]
    for i in xrange(M):
        start=time.time()
        match(
            matrix_gpu_left,
            matrix_gpu_right,
            output_gpu_disp,
            grid = ((WIDTH-1)/TILE_WIDTH+1,(HEIGHT-1)/TILE_WIDTH+1,1),
            block = (TILE_X,TILE_Y,1),
            )
        times.append(time.time()-start)
    tile_time=np.average(times)
    #print 'SAD with tile running time is:   '
    print tile_time
    A = output_gpu_disp.get()
    print A.tolist()
    # Regularization for 'smooth_55' and 'blur'
    A=A*255/DisparityLevel

    # A = output_gpu.get()
    Image.fromarray(A).convert('RGB').save('depth.png')
    # source = cv2.imread("depth.png",CV_LOAD_IMAGE_GRAYSCALE)
    # final=cv2.medianBlur(source,3)
    # final=Image.fromarray(final)

    # final.convert('RGB').save("depth.png")

#define a function to convert image to array
def create_img(filename):
    im = Image.open(filename).convert('L')
    return np.array(im)

#Calculate Running Time
M=5
times=[]
for i in xrange(M):
    start=time.time()
    imageConv('imL.png','imR.png')
    times.append(time.time()-start)
tile_time=np.average(times)
print 'SAD with tile running time is:   '
print tile_time
#imageConv('imL.png','imR.png')
