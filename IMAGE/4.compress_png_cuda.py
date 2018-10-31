#coding:utf-8

from PIL import Image
import numpy as np
from numpy import *  
import os
import sys
import pycuda.autoinit
import pycuda.driver as cuda
from timeit import default_timer as timer
from pycuda.compiler import SourceModule


 

def cuda_pooling(target_size, path, img_name):
    img=Image.open(path+img_name)
  
    prior_wid, prior_hei = img.size[0],img.size[1]

    width,height = target_size[0], target_size[1]
   
    window = int(prior_wid/width)

   
    a = np.array(img,dtype=np.int64)
    b  = np.ones((height,width,3),dtype=np.int64)
    d = np.array([window],dtype=np.int64)
    arr_len = np.array([30000],dtype=np.int64)

   
    a_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize)
    b_gpu = cuda.mem_alloc(b.size * b.dtype.itemsize)
    d_gpu = cuda.mem_alloc(d.size * d.dtype.itemsize)
    arr_len_gpu = cuda.mem_alloc(arr_len.size * arr_len.dtype.itemsize)

    cuda.memcpy_htod(a_gpu, a)
    cuda.memcpy_htod(b_gpu, b)
    cuda.memcpy_htod(d_gpu, d)
    cuda.memcpy_htod(arr_len_gpu, arr_len)

    mod = SourceModule("""
    __global__ void func(int *a, int *b,int* win, int * arr_len)
    {
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < arr_len[0])

        {
            
                int i=0;
                int min_pix = 765;
                int count_white=0;
                int begin = idx * win[0] * 3 ;
                for(i=0; i<win[0]; i++) 
                {

                    if( a[begin + i*3 ]+a[begin + i*3 +1 ]+a[begin + i*3 +2] < min_pix)
                    {
                        b[idx*3]=a[begin + i*3 ];
                        b[idx*3+1]=a[begin + i*3 +1];
                        b[idx*3+2]=a[begin + i*3 +2];
                        min_pix = a[begin + i*3 ]+a[begin + i*3 +1 ]+a[begin + i*3 +2];
                    }
                    if( a[begin + i*3 ]==255 && a[begin + i*3 +1 ]==255 && a[begin + i*3 +2]==255)
                    {
                        count_white+=1;
                    }            
                }
                if(count_white > (win[0]/2))
                {
                    b[idx*3] = 255;
                    b[idx*3+1]= 255;
                    b[idx*3+2] = 255;
                } 

            idx += blockDim.x * gridDim.x;
        }
    }
    """)
    func = mod.get_function("func")  

    nTheads = 1000
    nBlocks = 10

    start = timer()
    func( a_gpu, b_gpu, d_gpu, arr_len_gpu, block=( nTheads, 1, 1 ), grid=( nBlocks,1 ) )

    
    cuda.memcpy_dtoh(b,b_gpu)
    cuda.memcpy_dtoh(a,a_gpu)

    run_time = timer() - start  

    new_img = Image.fromarray(b.astype('uint8'), 'RGB')
    new_img.save(''+img_name.strip('.png')+'.cuda.png', "PNG")


def pooling(target_size, path, img_name):
    
    img=Image.open(path+img_name)
    pixel_array = np.array(img)
    
    prior_wid, prior_hei = img.size[0],img.size[1]
    width,height = target_size[0], target_size[1]
    
    window = int(prior_wid/width)
    newim = Image.new("RGB", (width, height), (255, 255, 255))

    for h in range(prior_hei):
        count=0
        new_colomn = -1
        box = []
        for w in range(prior_wid):
            count+=1
            if count == window:
                new_colomn += 1
                
                count_white=0
                for i in box:
                    if sum(i) < sum(newim.getpixel((new_colomn, h))):
                        newim.putpixel((new_colomn, h), (i[0], i[1], i[2]))
                    if i[0]==255 and i[1]==255 and i[2]==255:
                        count_white+=1
                if count_white > window/2:
                    newim.putpixel((new_colomn, h), (255, 255, 255))

                count = 0
                box = []
            elif count < window:
                box.append(pixel_array[h][w])

    newim.save(''+img_name.strip('.png')+'.cpu.png', "PNG")



def main():
    img_path = sys.argv[1]
    target_size = (,)
    start = timer()
    for file in os.listdir(img_path):
        if '.png' in file:
            cuda_pooling(target_size, img_path, file)
            #pooling(target_size, img_path, file)
    print("Finally, gpu run time %f seconds " % (timer() - start ))

if __name__ == '__main__':
    main()
