import matplotlib.patches as mpatches
import cv2
import numpy as np
import skimage
from skimage import util,segmentation,exposure,filters, morphology,measure,feature,io,draw
from scipy import ndimage,stats,cluster,misc,spatial
from sklearn.cluster import KMeans
from sklearn.neighbors  import NearestNeighbors

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from math import sqrt



def LoG_seed_detection (IMG, blob_LoG_Para = [14,22,35,0.01,0.7]): #LoG_Para = [14,22,35,0.01,0.7]
    blobRadius_min = blob_LoG_Para[0];
    blobRadius_max = blob_LoG_Para[1];
    num_sigma      = blob_LoG_Para[2];
    blob_thres     = blob_LoG_Para[3];
    overlap        = blob_LoG_Para[4];
    blob_radius_range_pixel = np.array([blobRadius_min, blobRadius_max])
    blob_radius_range = blob_radius_range_pixel /1.414                                    #  radius approximate root 2 * sigma 
    blobs_LoG  = feature.blob_log (
                        IMG, min_sigma = blob_radius_range[0], max_sigma = blob_radius_range[1] , 
                        num_sigma = num_sigma,                                           # number of sigma consider between the range
                        threshold = blob_thres , overlap= overlap
                       )     
#    print('LoG_seed_detection done  LoG_Paras are: ', blob_LoG_Para)
    return blobs_LoG

def GenerateSeeds_marker(IMG,blobs,diskR = 3):   # start with 1.2...   
    seed_centroidImg = np.zeros_like(IMG)
    for i,(x,y) in enumerate( zip( np.uint(blobs[:,0]), np.uint(blobs[:,1]) ) ):         #blobs read from seed detection result (blobs_log) or seed table
        seed_centroidImg[x,y] = (i+1)                    # disks of seeds are label as their id (1,2,3....)
    seeds_marker = morphology.dilation (seed_centroidImg,morphology.disk(diskR))         # sure forground (marked) is from blobs with same radius

    return seeds_marker

def GenerateBlob_marker(IMG,blobs):   # start with 1.2...    create muiltiple circle with x,y,r of blob[0,1,2] 

        
    Blob_marker = np.zeros_like(IMG)
    
    x, y = np.indices(IMG.shape)
    for i in  range (blobs.shape[0]):    
        x_i = blobs[i,0]
        y_i = blobs[i,1]
        r_i =(blobs[i,2]) 
#        print (r_i)
#        r_i_adjusted = r_i * 3.6/np.log(r_i) 
        r_i_adjusted = r_i  
#        print (r_i_adjusted)

#        rr, cc = skimage.draw.circle( *list(blob[i,:]))
        rr, cc = skimage.draw.circle( x_i,y_i,r_i_adjusted)
        r_bdId =  rr< IMG.shape[0]                             # fixin the boundary of image
        c_bdId =  cc< IMG.shape[1] 
        bdId = np.logical_and(r_bdId,c_bdId)       # only keep the ids within boundary

        Blob_marker[rr[bdId] , cc[bdId] ] = 1

    return Blob_marker

##''' input image : DAPI + HISTONE (pixel summation) '''   
def imSummation (img1,img2, outputformat = '16bit') :  # save size  
    imSum = np.add(img1, img2)
    imSum = imSum/ imSum.max()                  # float    
    
    if outputformat ==  '16bit':
        imSum = skimage.img_as_uint(imSum)   # change to 16bit 
        
    return imSum


def binMaskCorrection(img, thres_value):
    bin_mask = img > thres_value                             # generate binary mask from original image
    bin_mask = morphology.binary_dilation (bin_mask,morphology.disk(3))                        
#    bin_mask = morphology.binary_opening (bin_mask,morphology.disk(3))                 # remove white noise for
    bin_mask = morphology.binary_closing(bin_mask,morphology.disk(3))                   # remove dark noise for
    bin_mask = ndimage.binary_fill_holes(bin_mask, morphology.disk(2))                  # filling holes
    bin_mask = morphology.binary_closing (bin_mask,morphology.disk(3))                        
    
    bin_mask = morphology.binary_erosion (bin_mask,morphology.disk(1))                        

    return bin_mask

def borderCorrection( bin_mask_border, maskCorrectR = 2, noiseR = 5):                 # need shape correction  correcti to round like
    bin_mask_border = morphology.binary_dilation  (bin_mask_border,morphology.disk(noiseR))       # # remove dark noise for

    bin_mask_border = morphology.binary_closing  (bin_mask_border,morphology.disk(maskCorrectR))       # # remove dark noise for
    bin_mask_border = ndimage.binary_fill_holes  (bin_mask_border,morphology.disk(maskCorrectR))       # filling holes
#                bin_mask_border = morphology.binary_closing  (bin_mask_border,morphology.disk(maskCorrectR))       # # remove dark noise for
    bin_mask_border = morphology.binary_opening  (bin_mask_border,morphology.disk(noiseR))                # # remove white noise for
    
    bin_mask_border = morphology.binary_erosion  (bin_mask_border,morphology.disk(noiseR))       
    
    return bin_mask_border


def fillinhole(img, secondRd = False ,binaryPar = 0.6):  # find the hole and fill in 
    img_fl = skimage.img_as_float(img)
    otsu_thres = filters.threshold_otsu(img_fl)
    bin_mask_level1 = img_fl>binaryPar*otsu_thres                                     # get hole : binaryPar = 0.6 for DAPI+HISTONE, 0.4 for DAPI

    bin_mask_level1 = morphology.closing(bin_mask_level1,morphology.disk(2))     # close the border
#    plt.figure(), plt.imshow(bin_mask_level1 ,cmap = 'gray') ,plt.title('bin_mask_level1')
    
    bin_mask_filled = ndimage.binary_fill_holes(bin_mask_level1, morphology.disk(2)  )                # filling holes
    bin_maks_holes =  np.array( bin_mask_filled *1 - bin_mask_level1 *1, dtype = np.bool) *1                       #y= a- b   
    bin_maks_holes =  morphology.binary_dilation (bin_maks_holes,morphology.disk(4)  )             # enlarger the hole to generate mask

    
    # complemtory of bin maks find samll dots
    bin_mask_level1_sp = morphology.dilation(bin_mask_level1,morphology.disk(1))     # close the border    
    bin_mask_filled_sp = ndimage.binary_fill_holes(bin_mask_level1_sp, morphology.disk(2)  )                # filling holes
    bin_maks_holes_sp =  np.array( bin_mask_filled_sp *1  - bin_mask_level1_sp *1  , dtype = np.bool) *1                       #y= a- b   
    bin_maks_holes_sp =  morphology.binary_dilation (bin_maks_holes_sp,morphology.disk(4)  )             # enlarger the hole to generate mask

    bin_maks_holes = np.logical_or( bin_maks_holes , bin_maks_holes_sp)
    
    if bin_maks_holes.sum() == 'nan' or  bin_maks_holes.sum() == 0:
        filledCell = img   # no hole detected
#        print(bin_maks_holes.sum())
#        plt.figure(),plt.imshow(filledCell,cmap ='gray'),plt.title('filled image ')        
    else:       
        fill_in_pixcel = morphology.dilation(img_fl,morphology.disk(16)) * bin_maks_holes
#        plt.figure(),plt.imshow(fill_in_pixcel,cmap ='gray'),plt.title('filled image ')        

        fill_in_pixced_smoothed = fill_in_pixcel[fill_in_pixcel>0].mean() + filters.gaussian(fill_in_pixcel, sigma=3)
#        plt.figure(), plt.imshow(fill_in_pixced_smoothed ,cmap = 'gray') ,plt.title('fill_in_pixced_smoothed')    
     # #################                      seconde round fill in   # for those hole are too big
        if secondRd== True:
#            print(fill_in_pixced_smoothed.max())

            otsu_thres_2nd = filters.threshold_otsu(fill_in_pixced_smoothed)
            bin_mask_level2 = fill_in_pixced_smoothed >1.2*otsu_thres_2nd
            bin_mask_level2_filled = ndimage.binary_fill_holes(bin_mask_level2, morphology.disk(2)  )                # filling holes
            bin_mask_level2_holes =  np.array(bin_mask_level2_filled *1 - bin_mask_level2*1, dtype = np.bool) *1                   # find the remaining holes
            bin_mask_level2_holes = morphology.binary_dilation (bin_mask_level2_holes,morphology.disk(4)  )       
            
            if bin_mask_level2_holes.max() > 0 :                
                fill_in_pixcel_level2 = morphology.dilation(fill_in_pixced_smoothed,morphology.disk(11)) * bin_mask_level2_holes                    
                fill_in_pixced_level2_smoothed = fill_in_pixcel_level2[fill_in_pixcel_level2>0].mean() + filters.gaussian(fill_in_pixcel_level2, sigma=2)                    
                fill_in_pixced_smoothed = fill_in_pixced_smoothed +  fill_in_pixced_level2_smoothed                                                    

        filledCell = imSummation(img_fl, fill_in_pixced_smoothed, outputformat = '16bit')
#        plt.figure(), plt.imshow(filledCell ,cmap = 'gray') ,plt.title('filledCell')                   
    
    return filledCell