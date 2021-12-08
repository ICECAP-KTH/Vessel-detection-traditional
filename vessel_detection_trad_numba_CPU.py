import cv2
import numpy as np
import math
import tifffile as tiff
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from numba import njit
import time


def post_processing(org,image,x_cutted):
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
   plt.show()

@njit  
def mean(data):
    sum=0
    for k in data:
        sum+=k
    return sum/len(data)

@njit
def variance(data, mean,ddof=0):
     n = len(data)
     res=0
     for x in data:
        res+=((x - mean) ** 2)
     return res / (n - ddof)

@njit
def stdev(data,mean):
    var = variance(data,mean)
    std_dev = math.sqrt(var)
    return std_dev
   

# Apply the CFAR filter. The CFAR filter implemented here is described in:
#   M. Lundberg, L. M. H. Ulander, W. E. Pierson, A. Gustavsson,
#   "A challenge problem for detection of targets in foliage," 
#   Proc. SPIE 6237, Algorithms for Synthetic Aperture Radar Imagery XIII,
#   62370K (17 May 2006); https://doi.org/10.1117/12.663594
@njit
def cfar(data):
    
    outer_kernel=31
    inner_kernel=19
    [rows,cols]=data.shape
    Ic=np.zeros((rows,cols))
    sz1=int(((outer_kernel-1)/2))
    sz2=int(((outer_kernel-inner_kernel)/2))
    Itw=np.zeros((rows+outer_kernel-1,cols+outer_kernel-1))
    Itw[sz1:sz1+rows,sz1:sz1+cols]=data
    
    window=np.ones((outer_kernel, outer_kernel))
    window[sz2:sz2+inner_kernel , sz2:sz2+inner_kernel]=0

    #indices of element equal to one
    [ys,xs]=np.nonzero(window)
    filter_cfar=np.zeros((outer_kernel,outer_kernel))
    #np.std(data)
    for i in range(rows):
        for j in range(cols):
            la=[]
            filter_cfar=Itw[i:i+outer_kernel,j:j+outer_kernel]
            for (x,y) in zip(xs,ys):
                la.append(filter_cfar[y, x])
            m=mean(la)    
            temp=data[i,j]-m
            if(temp!=0.0):
               x_temp=(temp/stdev(la,m))
               if(x_temp>6):
                    Ic[i,j]=1


    return Ic

t0 = time.time()

# SELECT 3 for NIR when u ahve tiff images, in tiff.imread it is RGBNIR
# SELECT 2(red) or 0(b) for the best results when u have png images with 3 channels
#cv2.imread() reads the image as BGR
band = 3


#Tiff image with 4 bands
img = tiff.imread("area2_20200717_094217_1040_3B_AnalyticMS_SR.tif")

#img = cv2.imread("m0001.png")

#select the band (either R,G,B,NIR)
band = img[:,:,band]

#In case of using tiff images u have to normalize it
NIR_band_norm_img = cv2.normalize(band, dst=None, alpha=0, beta=65553, norm_type=cv2.NORM_MINMAX)

#In case u have land in the image crop it
x_cutted=499

#select only the region of water in Area
NIR_band_cutted = NIR_band_norm_img[:,x_cutted:]
#print(type(NIR_band_cutted[0][1]))

#call CFAR filter
Ic=cfar(NIR_band_cutted)


# Creating kernel
kernel = np.ones((3, 3), np.uint8)
image = cv2.erode(Ic, kernel)
img_final = cv2.dilate(image, kernel)


t1 = time.time()    
print(float(t1-t0))

post_processing(band,img_final,x_cutted)

 

   
