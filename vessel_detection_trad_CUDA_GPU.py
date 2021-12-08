import cv2
import numpy as np
import math
import tifffile as tiff
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from numba import njit,cuda
import time
import os


def post_processing(org,image,x_cutted,filename):
   fig, ax = plt.subplots()
   ax.imshow(org)
   label_img = label(image)
   regions = regionprops(label_img)
   i=0
   plt.show
   for props in regions:
        i=i+1
        y0, x0 = props.centroid
        orientation = props.orientation
        minr, minc, maxr, maxc = props.bbox
        diagonal=np.sqrt(np.power((maxr-minr),2) + np.power((maxc-minc),2))
        bx = (minc+x_cutted, maxc+x_cutted, maxc+x_cutted, minc+x_cutted, minc+x_cutted)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, '-b', linewidth=1.5)
        print("Diagonal: ",int(diagonal)," [",maxc-minc,",",maxr-minr,"], Orientation: ",int(np.degrees(orientation)),"Centroid: (",int(y0), int(x0+x_cutted),")")
   print(i)
   ax.axis((0, len(org[0]), len(org), 0))
   plt.savefig(Output_directory+filename+"_labelled.png")
   #plt.show()


@njit  
def mean(data):
    sum=0
    for k in data:
        sum+=k
    return sum/len(data)

@njit
def variance(data, mean):
     n = len(data)
     res=0
     for x in data:
        res+=(x - mean)*(x - mean)
     return res / n
     

@njit
def stdev(data,mean):
    var = variance(data,mean)
    std_dev = math.sqrt(var)
    return std_dev
   
@njit
def mat(rows,cols,val):
    data=[]
    for i in range(rows):
        for j in range(cols):
            data[i,j]=val
    return data


# Apply the CFAR filter. The CFAR filter implemented here is described in:
#   M. Lundberg, L. M. H. Ulander, W. E. Pierson, A. Gustavsson,
#   "A challenge problem for detection of targets in foliage," 
#   Proc. SPIE 6237, Algorithms for Synthetic Aperture Radar Imagery XIII,
#   62370K (17 May 2006); https://doi.org/10.1117/12.663594
@cuda.jit
def cfar(rows,cols,data,Itw,xs,ys,filter_cfar,la,outer_kernel,Ic):
    xcount,ycount=cuda.grid(2)
    if xcount<rows and ycount<cols:
            f=0
            filter_cfar=Itw[xcount:xcount+outer_kernel,ycount:ycount+outer_kernel]
            for (x,y) in zip(xs,ys):
                la[f]=(filter_cfar[y, x])
                f=f+1
            m=mean(la)
            temp=data[xcount,ycount]-m
            if(temp!=0.0):
               x_temp=(temp/stdev(la,m))
               if(x_temp>6):
                    Ic[xcount,ycount]=1
    
def start(filename):
    
    t0 = time.time()
    # SELECT 3 for NIR when u ahve tiff images, in tiff.imread it is RGBNIR
    # SELECT 2(red) or 0(b) for the best results when u have png images with 3 channels
    #cv2.imread() reads the image as BGR
    band = 3


    #Tiff image with 4 bands
    img = tiff.imread(Input_directory+filename)
    
    #img = cv2.imread(Input_directory+filename)

    #select the band (either R,G,B,NIR)
    band_selected = img[:,:,band]

    #In case of using tiff images u have to normalize it
    NIR_band_norm_img = cv2.normalize(band_selected, dst=None, alpha=0, beta=65553, norm_type=cv2.NORM_MINMAX)

    #In case u have land region in the image crop it , 600 is number to use to cut the left half of (1200x1200) image 
    x_cutted=499

    #select only the region of water in Area
    NIR_band_cutted = NIR_band_norm_img[:,x_cutted:]


    outer_kernel=31
    inner_kernel=19

    #CASE OF TIFF IMAGES
    [rows,cols]=NIR_band_cutted.shape

    #[rows,cols]=band_selected.shape
    Ic=np.zeros((rows,cols))
    sz1=int(((outer_kernel-1)/2))
    sz2=int(((outer_kernel-inner_kernel)/2))
    Itw=np.zeros((rows+outer_kernel-1,cols+outer_kernel-1))

    #Itw[sz1:sz1+rows,sz1:sz1+cols]=band_selected

    #CASE OF TIFF IMAGES
    Itw[sz1:sz1+rows,sz1:sz1+cols]=NIR_band_cutted

    window=np.ones((outer_kernel, outer_kernel))
    window[sz2:sz2+inner_kernel , sz2:sz2+inner_kernel]=0

    #indices of element equal to one
    [ys,xs]=np.nonzero(window)

    filter_cfar=np.zeros((outer_kernel,outer_kernel))
    la=np.zeros(len(ys))

    #call CFAR filter
    threadsperblock = (8, 8)
    #CASE OF TIFF IMAGES
    blockspergrid_x = math.ceil(NIR_band_cutted.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(NIR_band_cutted.shape[1] / threadsperblock[1])

    #blockspergrid_x = math.ceil(band_selected.shape[0] / threadsperblock[0])
    #blockspergrid_y = math.ceil(band_selected.shape[1] / threadsperblock[1])

    blockspergrid = (blockspergrid_x, blockspergrid_y)
     #CASE OF TIFF IMAGES
    cfar[blockspergrid,threadsperblock](rows,cols,np.uint32(NIR_band_cutted),Itw,np.uint16(xs),np.uint16(ys),filter_cfar,la,outer_kernel,Ic)
    #cfar[blockspergrid,threadsperblock](rows,cols,np.uint32(band_selected),Itw,np.uint16(xs),np.uint16(ys),filter_cfar,la,outer_kernel,Ic)

    # Creating kernel
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.erode(Ic, kernel)
    img_final = cv2.dilate(image, kernel)

    t1 = time.time()    
    print(float(t1-t0)) 

    post_processing(band_selected,img_final,x_cutted,filename)

    return float(t1-t0)




#Input_directory = 'MASATI-v2/images/'
#Output_directory = 'MASATI-v2/labelled/'

#CASE OF TIFF IMAGES
Input_directory = 'Unibap-Dataset/images/'
Output_directory = 'Unibap-Dataset/labelled/'

i=0
tot_time=0
for filename in os.listdir(Input_directory):
    print(filename)
    tot_time+=start(filename)
    i+=1


print("Total Time: "+str(tot_time/i)) 


   
