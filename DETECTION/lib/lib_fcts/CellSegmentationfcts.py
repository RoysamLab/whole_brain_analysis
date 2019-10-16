# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:59:38 2017

@author: xli63
"""

import os
#os.chdir(r'D:\research in lab\NIHIntern(new)\RebeccaCode')  # set current working directory
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import numpy as np
import skimage
from skimage import util,segmentation,exposure,filters, morphology,measure,feature,io
from scipy import ndimage,stats,cluster,misc,spatial
from sklearn.cluster import KMeans
from sklearn.neighbors  import NearestNeighbors

import numpy as np
import cv2
import heapq
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from math import sqrt



def LoG_seed_detection (IMG, blob_LoG_Para):
    blobRadius_min = blob_LoG_Para[0];
    blobRadius_max = blob_LoG_Para[1];
    num_sigma      = blob_LoG_Para[2];
    blob_thres     = blob_LoG_Para[3];
    overlap        = blob_LoG_Para[4];
    blob_radius_range_pixel = np.array([blobRadius_min, blobRadius_max])
    blob_radius_range = blob_radius_range_pixel /1.414                                    #  radius approximate root 2 * sigma 
    #plt.figure(),plt.imshow(image_masked,cmap='gray')
    blobs_LoG  = feature.blob_log (
                        IMG, min_sigma = blob_radius_range[0], max_sigma = blob_radius_range[1] , 
                        num_sigma = num_sigma,                                           # number of sigma consider between the range
                        threshold = blob_thres , overlap= overlap
                       )     
    print('LoG_seed_detection done  LoG_Paras are: ', blob_LoG_Para)
    return blobs_LoG

def GenerateSeeds_marker(IMG,blobs, diskR = 3):   # start with 1.2...
    seed_centroidImg = np.zeros_like(IMG)
    for i,(x,y) in enumerate( zip( np.uint(blobs[:,0]), np.uint(blobs[:,1]) ) ):         #blobs read from seed detection result (blobs_log) or seed table
        seed_centroidImg[x,y] = (i+1)                    # disks of seeds are label as their id (1,2,3....)
    seeds_marker = morphology.dilation (seed_centroidImg,morphology.disk(diskR))         # sure forground (marked) is from blobs with same radius
    return seeds_marker

def binMaskCorrection(img, thres_value):
    bin_mask = img > thres_value                             # generate binary mask from original image
    bin_mask = morphology.binary_dilation (bin_mask,morphology.disk(3))                        
#    bin_mask = morphology.binary_opening (bin_mask,morphology.disk(3))                 # remove white noise for
    bin_mask = morphology.binary_closing(bin_mask,morphology.disk(3))                   # remove dark noise for
    bin_mask = ndimage.binary_fill_holes(bin_mask, morphology.disk(5))                  # filling holes
    bin_mask = morphology.binary_closing (bin_mask,morphology.disk(3))                        
    return bin_mask

def borderCorrection( bin_mask_border, maskCorrectR):                 # need shape correction 
    bin_mask_border = morphology.binary_dilation  (bin_mask_border,morphology.disk(5))       # # remove dark noise for

    bin_mask_border = morphology.binary_closing  (bin_mask_border,morphology.disk(maskCorrectR))       # # remove dark noise for
    bin_mask_border = ndimage.binary_fill_holes  (bin_mask_border,morphology.disk(maskCorrectR))       # filling holes
#                bin_mask_border = morphology.binary_closing  (bin_mask_border,morphology.disk(maskCorrectR))       # # remove dark noise for
    bin_mask_border = morphology.binary_opening  (bin_mask_border,morphology.disk(5))                # # remove white noise for
    
    bin_mask_border = morphology.binary_erosion  (bin_mask_border,morphology.disk(5))       
    
    return bin_mask_border

def imSummation (img1,img2, outputformat = '16bit') :  # save size  
    imSum = np.add(img1, img2)
    imSum = imSum/ imSum.max()                  # float
    
    if outputformat ==  '16bit':
        imSum = skimage.img_as_uint(imSum)   # change to 16bit 

    return imSum

def fillinhole(img_fl, secondRd = False):  # find the hole and fill in 

    otsu_thres = filters.threshold_otsu(img_fl)
    bin_mask_level1 = img_fl>0.73*otsu_thres                                          # get hole
#    plt.figure(), plt.imshow(bin_mask_level1 ,cmap = 'gray')

    bin_mask_level1 = morphology.dilation(bin_mask_level1,morphology.disk(3))     # close the border
    bin_mask_filled = ndimage.binary_fill_holes(bin_mask_level1, morphology.disk(3)  )                # filling holes
    bin_maks_holes =  np.array(bin_mask_filled  *1 -  bin_mask_level1*1, dtype = np.bool) *1                       #y= a- b   
    bin_maks_holes =  morphology.binary_dilation (bin_maks_holes,morphology.disk(4)  )             # enlarger the hole to generate mask

    
    # complemtory of bin maks
    bin_mask_level1_sp = morphology.dilation(bin_mask_level1,morphology.disk(2))     # close the border    
    bin_mask_filled_sp = ndimage.binary_fill_holes(bin_mask_level1_sp, morphology.disk(2)  )                # filling holes
    bin_maks_holes_sp =  np.array(bin_mask_filled_sp *1 - bin_mask_level1_sp *1 , dtype = np.bool) *1                       #y= a- b   
    bin_maks_holes_sp =  morphology.binary_dilation (bin_maks_holes_sp,morphology.disk(4)  )             # enlarger the hole to generate mask

    bin_maks_holes_level1 = np.logical_or( bin_maks_holes , bin_maks_holes_sp)
    
    fill_in_pixcel = morphology.dilation(img_fl,morphology.disk(8)) * bin_maks_holes_level1
#    print(fill_in_pixcel.sum())
#    fill_in_pixced_smoothed = fill_in_pixcel[fill_in_pixcel>0].mean() + filters.gaussian_filter(fill_in_pixcel, sigma=2)
  #    plt.figure(), plt.imshow(bin_maks_holes_level1 ,cmap = 'gray')

 # #################                      seconde round fill in   # for those hole are too big
    if secondRd== True and fill_in_pixcel.sum() >2 :
        otsu_thres_2nd = filters.threshold_otsu(fill_in_pixcel)
        bin_mask_level2 = fill_in_pixcel > 1.2*otsu_thres_2nd
        bin_mask_level2_filled = ndimage.binary_fill_holes(bin_mask_level2, morphology.disk(3)  )                # filling holes
        bin_mask_level2_holes =  np.array(bin_mask_level2 *1 - bin_mask_level2_filled *1 , dtype = np.bool) *1                   # find the remaining holes
     
        fill_in_pixcel_level2 = morphology.dilation(fill_in_pixcel,morphology.disk(1)) * bin_mask_level2_holes    
  
        fill_in_pixcel = fill_in_pixcel + fill_in_pixcel_level2
            
    fill_in_pixced_smoothed = filters.gaussian(fill_in_pixcel, sigma=2)

    filledCell = imSummation(img_fl, 0.6*fill_in_pixced_smoothed, outputformat = '16bit')
            
    return filledCell

def watershedSegmentation (img, blobs, maskCorrectR = 0, maskDilateR = 0, LoG_Para = [],Bootstrap = False  , offset = 0.15,fillhole = False):
#    plt.figure()
#    plt.imshow(img,cmap ='gray')
    #
    
    if fillhole == True:
        # change to the img: fill in holes on original images
        img = fillinhole(skimage.img_as_float(img), secondRd = True)  # find the hole and fill in 
        img = skimage.img_as_uint(img)

    otsu_thres = filters.threshold_li(img)
    bin_mask_level1 = binMaskCorrection(img, (1 - 0 ) * otsu_thres)
    
    if maskDilateR!=0:     # enlarge the border for small components
        bin_mask_level1 = morphology.binary_dilation  (bin_mask_level1,morphology.disk(maskDilateR))     
    if maskCorrectR!=0:   # fill in holes
        bin_mask_level1 = borderCorrection( bin_mask_level1, maskCorrectR)    
    
    otsu_thres = filters.threshold_otsu(img)
        
    bin_mask = binMaskCorrection(img, 0.8 * otsu_thres)
    
    bin_mask_shrinked = binMaskCorrection(img, otsu_thres)

#    img_Intensed = img + filters.laplace(img, ksize=10, mask=bin_mask)      # add laplace fileter to it more intensive

#    border_otsuMask = segmentation.find_boundaries(bin_mask)
    
#''' Improve the thresholding'''
##img_Intensed = img + filters.laplace(img, ksize=10)      # add laplace fileter to it more intensive
#adaptive_thresh = filters.threshold_triangle(util.invert(img), nbins=528)
#new = morphology.h_minima(img, h =1, selem= morphology.disk(1))
#new = morphology.binary_closing (new,morphology.disk(3))                      
#
#plt.figure(),plt.imshow(new , cmap= 'gray')
#
#plt.figure(),plt.imshow(img, cmap= 'gray')
#  _________
    
    D = ndimage.distance_transform_edt(bin_mask)                                         # generate distant map, centrois locates in peaks
    D_shrinked  = ndimage.distance_transform_edt(bin_mask_shrinked)  
    
    D = D + 5*D_shrinked                                                                  #!!!!!!!!!! correct the border shape
#    D = morphology.erosion(D,morphology.disk(3))
    
    #Generate sure foreground
    seeds_marker_1st = GenerateSeeds_marker (img, blobs)
    
    #in case it needs to generatre the whole cell mask 
    if maskDilateR != 0:
        bin_mask = morphology.binary_dilation (bin_mask, morphology.disk(maskDilateR)) 
    
    bin_mask = np.logical_or(bin_mask, (seeds_marker_1st>0) )                             # make sure the seeds marker has considered
    
#     Implement Watershed 
    labels_1st = morphology.watershed(-D, seeds_marker_1st, mask=bin_mask)                   # labeled components, background = 0, other= ID, with eact shape of blobs
    PropertyTable_1st = measure.regionprops(labels_1st,intensity_image=img)                  # storage the properties e.g ID,area  of each componentes (labelled regions)
#    
    # put into result
    seeds_marker = seeds_marker_1st
    labels = labels_1st
    PropertyTable = PropertyTable_1st         
    
    updated_blobs = blobs        
    
    if Bootstrap == True:  ## adjust the labels by itself   will change the number of cells!!
        Mask_2nd = np.zeros_like(bin_mask)
        #1) find the missing compoments
        missingmask = np.logical_xor(bin_mask,(labels_1st>0))
        missingmask_label = skimage.morphology.label(missingmask, neighbors=None, background=None, return_num=False, connectivity=None)  # Label connected regions of an integer array.
        for missingComponent in measure.regionprops(missingmask_label):
            if missingComponent.area > 100:  # the missing component is big enough
                Mask_2nd[missingmask_label == missingComponent.label] = 1           
#        #2) find the connected compoments (find more than one component labelled as one ID)
#        removelabels = []
#        for Component in PropertyTable:
#            if Component.area > 1500:                                                    # the connected component is big enough
#                Mask_2nd[labels == Component.label] = 1                                  # add the label to 2nd round analysis
#                
#                PropertyTable.remove(Component)                                          # remove the information from original propertyTable (! cautious, the ID won't be consecutive)
#                removelabels.append(Component.label)                                     # storage the label ID to be removed 
#                labels[labels == Component.label] = 0                                    # change the labelle componentes to be removed to  0 (background)
#                seeds_marker[labels == Component.label] = 0  
#                
##           print ('label ='+ str (Component.label)+' area='+ str( Component.area ))  
#            plt.figure(), plt.imshow(Mask_2nd)
#        border_otsuMask_2nd = segmentation.find_boundaries(Mask_2nd)         ###
#
        img_2nd = img * Mask_2nd
        '''redo blob_loG and watershed again for the masked image'''
#        D_2nd = ndimage.distance_transform_edt(Mask_2nd)                                 # generate distant map, centrois locates in peaks
#        D_2nd = morphology.erosion(D_2nd,morphology.disk(3))             
        blobs_2nd = LoG_seed_detection (img_2nd, LoG_Para)              
        updated_blobs = np.concatenate( (blobs, blobs_2nd), axis= 0 )        
        seeds_marker, labels, PropertyTable,D, updated_blobs  = watershedSegmentation (img, updated_blobs, maskDilateR, LoG_Para = [], Bootstrap = False )
        

        
    print('Use watershed generate segmentation borders done!')
    
    return seeds_marker, labels, PropertyTable,D, updated_blobs
