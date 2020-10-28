# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:59:38 2017

@author: xli63
"""
#LoG_Para = [14,22,35,0.01,0.7]

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

import sys
#sys.path.insert(0, os.getcwd()+'/lib_fcts')
#import CellSegmentationfcts as myfcts                                          # my functions
#from Dataset_read_TBI_BFcorrected import Dataset_read_TBI                                  # read file proterty files
#ds = Dataset_read_TBI()                                                            # define a class to read the input channels image(image name and paramaters)



def LoG_seed_detection (IMG, blob_LoG_Para):
    blobRadius_min = blob_LoG_Para[0]
    blobRadius_max = blob_LoG_Para[1]
    num_sigma      = blob_LoG_Para[2]
    blob_thres     = blob_LoG_Para[3]
    overlap        = blob_LoG_Para[4]
    blob_radius_range_pixel = np.array([blobRadius_min, blobRadius_max])
    blob_radius_range = blob_radius_range_pixel /1.414                                    #  radius approximate root 2 * sigma 
    blobs_LoG  = feature.blob_log (
                        IMG, min_sigma = blob_radius_range[0], max_sigma = blob_radius_range[1] , 
                        num_sigma = num_sigma,                                           # number of sigma consider between the range
                        threshold = blob_thres , overlap= overlap
                       )     
#    print('LoG_seed_detection done  LoG_Paras are: ', blob_LoG_Para)
    return blobs_LoG

def GenerateSeeds_marker(IMG_shape,blobs,diskR = 3):   # start with 1.2...   
    # Input: 
    #       gray scale IMG shape [width , heights]
    #       blobs:   N * 2 : coords of all seeds             [ [x1,y1],[x2,y2]...[xn,yn]]
    #            (or N * 3: coords and radius of all seeds  [ [x1,y1,r1],[x2,y2,r2]...[xn,yn,rn]] ) 
    # output:
    #      seeds_marker, the same size with input image 

    seed_centroidImg = np.zeros(IMG_shape,dtype= np.uint32)
    for i,(x,y) in enumerate( zip( np.uint(blobs[:,0]), np.uint(blobs[:,1]) ) ):         #blobs read from seed detection result (blobs_log) or seed table
        seed_centroidImg[x,y] = (i+1)                    # disks of seeds are label as their id (1,2,3....)
    seeds_marker = morphology.dilation (seed_centroidImg,morphology.disk(diskR))         # sure forground (marked) is from blobs with same radius

    return seeds_marker

def imSummation (img1,img2, outputformat = '16bit') :  # save size  
    imSum = np.add(img1, img2)
    imSum = imSum/ imSum.max()                  # float    
    
    if outputformat ==  '16bit':
        imSum = skimage.img_as_uint(imSum)   # change to 16bit 
        
    return imSum


def binMaskCorrection(img, thres_value):
    # import pdb; pdb.set_trace()
    bin_mask = img > thres_value                             # generate binary mask from original image
    bin_mask = morphology.binary_dilation (bin_mask,morphology.disk(3))                        
    #bin_mask = morphology.binary_opening (bin_mask,morphology.disk(3))                 # remove white noise for
    bin_mask = morphology.binary_closing(bin_mask,morphology.disk(3))                   # remove dark noise for
    bin_mask = ndimage.binary_fill_holes(bin_mask, morphology.disk(2))                  # filling holes
    bin_mask = morphology.binary_closing (bin_mask,morphology.disk(3))                        
    
    bin_mask = morphology.binary_erosion (bin_mask,morphology.disk(1))                        

    return bin_mask

def borderCorrection( bin_mask_border, maskCorrectR):                 # need shape correction 
    bin_mask_border = morphology.binary_dilation  (bin_mask_border,morphology.disk(5))       # # remove dark noise for

    bin_mask_border = morphology.binary_closing  (bin_mask_border,morphology.disk(maskCorrectR))       # # remove dark noise for
    bin_mask_border = ndimage.binary_fill_holes  (bin_mask_border,morphology.disk(maskCorrectR))       # filling holes
    # bin_mask_border = morphology.binary_closing  (bin_mask_border,morphology.disk(maskCorrectR))       # # remove dark noise for
    bin_mask_border = morphology.binary_opening  (bin_mask_border,morphology.disk(5))                # # remove white noise for
    
    bin_mask_border = morphology.binary_erosion  (bin_mask_border,morphology.disk(5))       
    
    return bin_mask_border


def fillinhole(img, secondRd = False ,binaryPar = 0.6):  # find the hole and fill in 
    img_fl = skimage.img_as_float(img)
    otsu_thres = filters.threshold_otsu(img_fl)
    bin_mask_level1 = img_fl>binaryPar*otsu_thres                                     # get hole : binaryPar = 0.6 for DAPI+HISTONE, 0.4 for DAPI

    bin_mask_level1 = morphology.closing(bin_mask_level1,morphology.disk(2))     # close the border
    #plt.figure(), plt.imshow(bin_mask_level1 ,cmap = 'gray') ,plt.title('bin_mask_level1')
    
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

    else:       
        fill_in_pixcel = morphology.dilation(img_fl,morphology.disk(16)) * bin_maks_holes

        fill_in_pixced_smoothed = fill_in_pixcel[fill_in_pixcel>0].mean() + filters.gaussian(fill_in_pixcel, sigma=3)
     # #################                      seconde round fill in   # for those hole are too big
        if secondRd== True:

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
    
    return filledCell

def watershedSegmentation( img, maskCorrectR = 0, maskDilateR = 0, LoG_Para = [],bgBoost = False  , offset = 0, imagefilling = False ,blobs =None):
    # blobs could either from outside or generated from loG in this function
    # the more offset, the more small element to be captured

    if imagefilling == True:
        img = fillinhole(img, secondRd = True)  # find the hole and fill in 
	
    img = exposure.rescale_intensity(skimage.img_as_float(img))    #   img adjsut update [20171008]

    otsu_thres = filters.threshold_otsu(img)
    # import pdb; pdb.set_trace()
    bin_mask_level1 = binMaskCorrection(img, (1 - offset ) * otsu_thres)

    #### Adjust cells
    if maskDilateR!=0:     # enlarge the border for small components
        bin_mask_level1 = morphology.binary_dilation  (bin_mask_level1,morphology.disk(maskDilateR))     
    if maskCorrectR!=0:   # fill in holes
        bin_mask_level1 = borderCorrection( bin_mask_level1, maskCorrectR)    
        
    ###################
    if blobs is not None :  # read blob from outside ,implememnt global watershed      +    
      
        seeds_marker = GenerateSeeds_marker (img.shape, blobs)    
        D             = ndimage.distance_transform_edt(bin_mask_level1)                                         # generate distant map, centroids  locates in peaks
        D_exactShape  = ndimage.distance_transform_edt(img>otsu_thres)          
        D = D + 5 * D_exactShape                                                                  #!!!!!!!!!! correct the border shape
        
        label_level1 = morphology.watershed(-D, seeds_marker,  mask = bin_mask_level1,compactness=0.8)                   # labeled components, background = 0, other= ID, with eact shape of blobs
        label_connected = np.zeros_like(label_level1)
        #Generate sure foreground
        #### within each crops
        connected_area = int(3.14 *  ( (LoG_Para[0] ) /2) **2 )
        smallest_area  = int(3.14 *  ( (LoG_Para[0]  )/3 ) **2 )

        ##### Step1 ,clean up the labels
        PropertyTable_1st = measure.regionprops(label_level1 )
        connected_obj_id = 1
        for connected_obj in PropertyTable_1st:
            if connected_obj.area < smallest_area : 
                # clean the labels
                label_level1[connected_obj.coords[:,0],connected_obj.coords[:,1]]  = 0               
                
            elif connected_obj.area > connected_area :   # size = 600
                # clean the labels
                label_level1[connected_obj.coords[:,0],connected_obj.coords[:,1]]  = 0        
                label_connected[connected_obj.coords[:,0],connected_obj.coords[:,1]] = connected_obj_id 
                connected_obj_id = connected_obj_id + 1
        ##### Step2 ,level2 segmentation  
        label_level2 = segmentation.relabel_sequential(label_level1 , offset=1)[0]
        label_level2_ID = label_level2.max()                                                # label of label_level2  should start with label_level1.max() 

        PropertyTable_1st = measure.regionprops(label_connected, intensity_image = img)
        for connected_obj in PropertyTable_1st:
            connected_Crop =    connected_obj.intensity_image           # size > 600
            # # clean the labels
            # label_level2[connected_obj.coords[:,0],connected_obj.coords[:,1]]  = 0                        
            # enlarger the window for crop
            enlarge_width = 5
            connected_Crop_enlarged = np.zeros( (connected_Crop.shape[0] + 2* enlarge_width , 
                                                    connected_Crop.shape[1] + 2* enlarge_width )  )
            connected_Crop_enlarged[enlarge_width : enlarge_width + connected_Crop.shape[0],
                                    enlarge_width : enlarge_width + connected_Crop.shape[1]] = connected_Crop
                    
            # correct he blog                
            otsu_thres_Crop = filters.threshold_otsu(connected_Crop_enlarged)                
            
            #  bin_mask_border       = binMaskCorrection(connected_Crop_enlarged,  (1 - offset ) * otsu_thres) 
            bin_mask_border       = binMaskCorrection(connected_Crop_enlarged,  (1 + offset ) * otsu_thres_Crop) 
            

            if maskCorrectR!=0:   # fill in holes
                bin_mask_border = morphology.binary_dilation  (bin_mask_border,morphology.disk(2))       # # remove dark noise for                
                bin_mask_border = morphology.binary_closing  (bin_mask_border,morphology.disk(2))       # # remove dark noise for
                bin_mask_border = ndimage.binary_fill_holes  (bin_mask_border,morphology.disk(2))       # filling holes
                        
            bin_mask_exactShape   = connected_Crop_enlarged > (1+0.3)*otsu_thres_Crop        
                    
            # generate distant map, centroids  locates in peaks
            D_exactShape = ndimage.distance_transform_edt(bin_mask_exactShape)           
            
            # create seeds_marker_crop
            seeds_marker_crop = seeds_marker[connected_obj.bbox[0] : connected_obj.bbox[2],
                                                connected_obj.bbox[1] : connected_obj.bbox[3] ]
            seeds_marker_crop_enlarged = np.zeros_like(connected_Crop_enlarged).astype(np.uint64)
            seeds_marker_crop_enlarged[ enlarge_width : enlarge_width + connected_Crop.shape[0],
                                        enlarge_width : enlarge_width + connected_Crop.shape[1]] = seeds_marker_crop

            #######     Implement Watershed  ##########

            # labels_Crops_enlarged = morphology.watershed(-D_exactShape, seeds_marker_crop_enlarged,  mask = bin_mask_border,compactness=0.8)                   # labeled components, background = 0, other= ID, with eact shape of blobs , add compactness update [20171008]
            labels_Crops_enlarged = morphology.watershed(- connected_Crop_enlarged, seeds_marker_crop_enlarged,  mask = bin_mask_border,compactness = 0.8)                   # labeled components, background = 0, other= ID, with eact shape of blobs
                            
            labels_Crops = labels_Crops_enlarged[enlarge_width : enlarge_width + connected_Crop.shape[0],         # recover it back from enlarged image
                                                    enlarge_width : enlarge_width + connected_Crop.shape[1]]
            labels_Crops = labels_Crops * connected_obj.filled_image                                         # make sure it don't bleed ousider the level 1 crop
            
            # within the connected_crops
            obj_inCrop_areas = []
            for obj_inCrop in measure.regionprops(labels_Crops) :             # the fill the labels_crops in to whole label image
                for i in  range (obj_inCrop.coords.shape[0]):                # for each pixel in the crops
                    label_level2[ connected_obj.bbox[0] + obj_inCrop.coords[i,0],
                                  connected_obj.bbox[1] + obj_inCrop.coords[i,1] ]  = label_level2_ID + obj_inCrop.label       # the label should start with the largetst label 
                label_level2_ID = label_level2_ID + obj_inCrop.label         # add the current largest label  
                obj_inCrop_areas.append(obj_inCrop.area)
    
        print ("len(np.unique(label_level2)) = ",len(np.unique(label_level2)) )
        labels = segmentation.relabel_sequential(label_level2 , offset=1)[0]                                                   

        updated_blobs = np.array(blobs)        

        
    else:#  blobs will generated from loG,implement hierarchy watershed and LoG                , create blob    
        label_level1 = skimage.morphology.label(bin_mask_level1, neighbors=None, background=None, return_num=False, connectivity=None)  # Label connected regions of an integer array.
        label_level2 = label_level1        
        label_level2_ID = label_level1.max()                                                # label of label_level2  should start with label_level1.max() 
    
        #### within each crops
        connected_area = int(3.14 *  ( (LoG_Para[0] ) /2) **2 )
        smallest_area  = int(3.14 *  ( (LoG_Para[0]  )/3 ) **2 )
    
        PropertyTable_1st = measure.regionprops(label_level1,intensity_image=img)                  # storage the properties e.g ID,area  of each componentes (labelled regions)
        for connected_obj in PropertyTable_1st:
            if connected_obj.area < smallest_area : 
                # clean the labels
                label_level2[connected_obj.coords[:,0],connected_obj.coords[:,1]]  = 0               
                
            elif connected_obj.area > connected_area :   # size = 600
                
                connected_Crop =    connected_obj.intensity_image           # size = 600

                # clean the labels
                label_level2[connected_obj.coords[:,0],connected_obj.coords[:,1]]  = 0        
                
                # enlarger the window for crop
                enlarge_width = 5
                connected_Crop_enlarged = np.zeros( (connected_Crop.shape[0] + 2* enlarge_width , 
                                                     connected_Crop.shape[1] + 2* enlarge_width )  )
                connected_Crop_enlarged[enlarge_width : enlarge_width + connected_Crop.shape[0],
                                        enlarge_width : enlarge_width + connected_Crop.shape[1]] = connected_Crop
                       
                # correct he blog
                
                otsu_thres_Crop = filters.threshold_otsu(connected_Crop_enlarged)                
                
                #  bin_mask_border       = binMaskCorrection(connected_Crop_enlarged,  (1 - offset ) * otsu_thres) 
                bin_mask_border       = binMaskCorrection(connected_Crop_enlarged,  (1 + offset ) * otsu_thres_Crop) 

                if maskCorrectR!=0:   # fill in holes
                    bin_mask_border = morphology.binary_dilation  (bin_mask_border,morphology.disk(2))       # # remove dark noise for                
                    bin_mask_border = morphology.binary_closing  (bin_mask_border,morphology.disk(2))       # # remove dark noise for
                    bin_mask_border = ndimage.binary_fill_holes  (bin_mask_border,morphology.disk(2))       # filling holes
                            
                bin_mask_exactShape   = connected_Crop_enlarged > (1+0.3)*otsu_thres_Crop        
                
        
                # generate distant map, centroids  locates in peaks
                D_exactShape = ndimage.distance_transform_edt(bin_mask_exactShape)           
                
                if connected_obj.area > 8* connected_area and connected_obj.area < 20* connected_area :    # the connected components are too big , probabably Hippocampus region
                    LoG_Para_shrinked = [ LoG_Para[0]-3, LoG_Para[1]-4,LoG_Para[2], LoG_Para[3] ,LoG_Para[4] ]
                    Crop_blobs = LoG_seed_detection (D_exactShape,  LoG_Para_shrinked)          #   [12,23,35,0.01,0.7]
                
                if connected_obj.area > 100* connected_area :    # the connected components are too big , probabably Hippocampus region
                    #print('Size of Connected Compnet is : ',connected_obj.area)
                    LoG_Para_shrinked = [ 8.6, 9,5, LoG_Para[3] ,LoG_Para[4] ]
                    #print('3rdLOG PAras : ',LoG_Para_shrinked)
                    
                    Crop_blobs = LoG_seed_detection (D_exactShape,  LoG_Para_shrinked)          #   [12,23,35,0.01,0.7]

                else:
                    Crop_blobs = LoG_seed_detection (D_exactShape,  LoG_Para)          #   [7, 20, 35, 0.015, 0.5]  
                    

                seeds_marker_crop = GenerateSeeds_marker (connected_Crop_enlarged.shape, Crop_blobs)
    
                #######     Implement Watershed  ##########
                print ( "Implement Watershed" )
                labels_Crops_enlarged = morphology.watershed(-D_exactShape, seeds_marker_crop,  mask = bin_mask_border,compactness=0.8)                   # labeled components, background = 0, other= ID, with eact shape of blobs , add compactness update [20171008]
                # labels_Crops_enlarged = morphology.watershed(- compositeIMG, seeds_marker_crop,  mask = bin_mask_border)                   # labeled components, background = 0, other= ID, with eact shape of blobs
                               
                labels_Crops = labels_Crops_enlarged[enlarge_width : enlarge_width + connected_Crop.shape[0],         # recover it back from enlarged image
                                                     enlarge_width : enlarge_width + connected_Crop.shape[1]]
                labels_Crops = labels_Crops * connected_obj.filled_image                                         # make sure it don't bleed ousider the level 1 crop
              
               # within the connected_crops
                obj_inCrop_areas = []
                for obj_inCrop in measure.regionprops(labels_Crops) :             # the fill the labels_crops in to whole label image
                    for i in  range (obj_inCrop.coords.shape[0]):                # for each pixel in the crops
                        label_level2[ connected_obj.bbox[0] + obj_inCrop.coords[i,0],
                                      connected_obj.bbox[1] + obj_inCrop.coords[i,1] ]  = label_level2_ID + obj_inCrop.label       # the label should start with the largetst label 
                    label_level2_ID = label_level2_ID + obj_inCrop.label         # add the current largest label  
                    obj_inCrop_areas.append(obj_inCrop.area)
        
        labels = segmentation.relabel_sequential(label_level2 , offset=1)[0]
        
        if bgBoost == True:  ## adjust the labels by itself   will change the number of cells!!
            Mask_2nd = np.zeros_like(bin_mask_level1)
            #1) find the missing compoments
            missingmask = np.logical_xor(bin_mask_level1,(label_level2>0))
            missingmask_label = skimage.morphology.label(missingmask)   # Label connected regions of an integer array.
            for missingComponent in measure.regionprops(missingmask_label, intensity_image = (img * missingmask)):
                if ( missingComponent.area > 150 ) and (missingComponent.mean_intensity>5500):  # the missing component is big enough and bright enough
                    Mask_2nd[missingmask_label == missingComponent.label] = 1       
                    
            missingmask_label =  missingmask_label * Mask_2nd
            missingmask_label =  segmentation.relabel_sequential(missingmask_label , offset = label_level2.max())[0]
            labels = labels + missingmask_label         
       
        
        labels = segmentation.relabel_sequential(labels , offset=1)[0]
         
        seed_centroidImg = np.zeros_like(labels)
        PropertyTable = measure.regionprops(labels)          
        
        updated_blobs = []
        for obj in PropertyTable  :         #blobs read from seed detection result (blobs_log) or seed table
            x = np.uint(obj.centroid[0])
            y = np.uint(obj.centroid[1])
            r = obj.equivalent_diameter/2
            updated_blobs.append([x,y,r]) 
            seed_centroidImg[x,y] = obj.label                   # disks of seeds are label as their id (1,2,3....)
        seeds_marker = morphology.dilation (seed_centroidImg,morphology.disk(3))         # sure forground (marked) is from blobs with same radius
        
        updated_blobs = np.array(updated_blobs)        

        
    print('Use watershed generate segmentation borders done!')
    
    return seeds_marker, labels, updated_blobs

