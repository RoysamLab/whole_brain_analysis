# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:29:11 2019

@author: xli63
"""
import numpy as np
from skimage.feature import ORB, match_descriptors
from skimage.transform import ProjectiveTransform, SimilarityTransform, AffineTransform,PolynomialTransform,resize
from skimage import segmentation,measure,morphology   
from scipy import ndimage
import multiprocessing
from multiprocessing.pool import ThreadPool
import warnings
import time
import cv2
warnings.filterwarnings("ignore")

from sklearn.neighbors import DistanceMetric

def featureExtract_tiled(img, paras, tileRange_ls, verbose = False):
    # verbose=True
    if paras.multiprocess == False:  # use single process option
        keypoint_all, descriptor_all = orb_detect_tiled(img, paras, tileRange_ls)

    else:                              # use multiprocess option
        # Set numOfThreads
        if str(paras.multiprocess).isnumeric():
            numOfThreads = int(paras.multiprocess)
            
        else:
            numOfThreads = multiprocessing.cpu_count()

        print ("\n numOfThreads=", str(numOfThreads))
        pool = ThreadPool(processes=numOfThreads)
        ls_size = int(np.ceil(len(tileRange_ls)/numOfThreads))
        async_result = []
        # run multiprocess 
        for th in range (0, numOfThreads):
            tileRange_ls_mp   = tileRange_ls[  th*ls_size: th*ls_size +ls_size ]     # split the whole tileRange_ls in to parts for multiprocessing  
            if len(tileRange_ls_mp) > 0 : 
                async_result.append(  
                        pool.apply_async( 
                                orb_detect_tiled, ( 
                                        img, paras, tileRange_ls_mp
                                    )))  # tuple of args for foo
                if verbose:
                    print("\tmulti thread for", th, " ... ,","len(tileRange_ls_mp)=",len(tileRange_ls_mp))
        pool.close()        
        pool.join()
        # load results
        keypoint_alltiles   = np.zeros((1,2))
        # descriptor_alltiles = np.zeros((1,32))   # skimage for 256, cv2 for 32
        descriptor_alltiles = np.zeros((1, 32)) if paras.keypoint_tool == "cv2" else np.zeros((1, 256))     # cv2, 8bit  32


        for r_i, r in enumerate( async_result):
            if verbose:
                print ("\tr_i=", r_i)

            keypoint_alltiles   = np.concatenate((keypoint_alltiles,   r.get()[0]), axis=0)
            descriptor_alltiles = np.concatenate((descriptor_alltiles, r.get()[1]), axis=0)
        # import pdb;pdb.set_trace()

        keypoint_all   =  keypoint_alltiles[1:, :]      # N by 2
        descriptor_all =  descriptor_alltiles[1:, :]    # N by 256
        # print("[multiprocessing] featureExtract_tiled: keypoint_alltiles.shape = ", keypoint_alltiles.shape)

    print("*" * 10 + "Detected keypoint numbers: = ", keypoint_all.shape[0] )

    return keypoint_all,descriptor_all

def orb_detect_tiled(img, paras, tileRange_ls,verbose = False ):
    ''' featureExtraction on large images with similar texture distribution '''
    # In order to feature extraction over all regions in the image, we will apply feature extraction tiled by tile
    # crop image into tiles and extract keypointss
    keypoint_alltiles   = np.zeros((1, 2))
    descriptor_alltiles = np.zeros((1, 32)) if paras.keypoint_tool == "cv2" else np.zeros((1, 256))     # cv2, 8bit  32
    if verbose:
        print ("$"*5 + "[Debug point 0]" )
        print ("len(tileRange_ls)=", len(tileRange_ls))

    for t_i, tileRange in enumerate( tileRange_ls):
        crop_img = img[ tileRange[0]:tileRange[2],   #i: i+crop_height
                        tileRange[1]:tileRange[3]]   #j: j + crop_width     
        if  crop_img.max() >0 and min(crop_img.shape)>1:              
            # only tiles contain enough engergy (obvious contrast of adjacent pixels) have keypoints                
            tile_n_keypoints = int(paras.n_keypoints / paras.tiles_numbers)  # past average distribute
            
            #compute the descriptors with ORB
            
            if paras.keypoint_tool == "skimage":
                
                orb = ORB(n_keypoints = tile_n_keypoints,
                        fast_threshold = paras.fast_threshold,
                        harris_k = paras.harris_k)
                #  if without this step, orb.detect_and_extract will return error
                orb.detect(crop_img)
                if verbose:
                    print ("$"*5 + "[Debug point 2]" )
                    print ("\tPre check whether keypoints are detected or not")
                    print ("\tlen(orb.scales)= " , len(orb.scales))

                if len(orb.scales) > 0:
                    orb.detect_and_extract(crop_img)
                    keypoints0 = orb.keypoints  # n by 2 array, n is number of keypoints
                    keypoint = keypoints0 + np.tile([tileRange[0], tileRange[1]],
                                                    (keypoints0.shape[0], 1))           # add the crop coordinate shift
                    descriptor = orb.descriptors                                        # n by 256 array
                    
                    if verbose:
                        print("\t\t" ,tileRange, "orb_detect_tiled : = ", keypoint.shape[0])

                    keypoint_alltiles   = np.concatenate((keypoint_alltiles  , keypoint  ), axis=0)
                    descriptor_alltiles = np.concatenate((descriptor_alltiles, descriptor), axis=0)

            else:# paras.keypoint_tool == "cv2":

                orb = cv2.ORB_create( nfeatures=paras.n_keypoints, scoreType=cv2.ORB_FAST_SCORE)
                kp, descriptor = orb.detectAndCompute(crop_img, None)
                if len(kp) > 0:
                    keypoints0 = cv2.KeyPoint_convert(kp)
                    keypoint = keypoints0 + np.tile([tileRange[0], tileRange[1]],
                                                    (len(keypoints0), 1))           # add the crop coordinate shift
                    if verbose:
                        print("\t" ,"t_i=",t_i,"shape=",crop_img.shape, "orb_detect_tiled : = ", keypoint.shape[0])
                    
                    keypoint_alltiles   = np.concatenate((keypoint_alltiles  , keypoint  ), axis=0)
                    descriptor_alltiles = np.concatenate((descriptor_alltiles, descriptor), axis=0)
                    

    keypoint_all    = keypoint_alltiles     [1:, :]
    descriptor_all  = descriptor_alltiles   [1:, :]
    

    return keypoint_all, descriptor_all


def match_descriptors_tiled (keypoints0,descriptors0,keypoints1,descriptors1,
                             target_shape, tileRange_ls ,ck_shift = 30,match_perc=1.0,
                             verbose = False):
    # print (" \n''' match_descriptors_tiled ")
    ck_tileRange_ls = []
    src   = np.zeros((1, 2))
    dst   = np.zeros((1, 2))
    for t_i, tileRange in enumerate( tileRange_ls):
        ''' Find the keypoints in the same tiles'''
        ck_tileRange=  [ max( 0 , tileRange[0] - ck_shift ), 
                         max( 0 , tileRange[1] - ck_shift ),
                         min( int(target_shape[0] ) -1, tileRange[2] + ck_shift) , 
                         min( int(target_shape[1] ) -1, tileRange[3] + ck_shift) ]         # extract the labels of subpicious window from previous merged results
        ck_tileRange_ls.append(ck_tileRange)                    
        bin_descriptors1  = ( keypoints1[:,0] >= tileRange[0] ) * ( keypoints1[:,0] <= tileRange[2] )
        bin_descriptors1 *= ( keypoints1[:,1] >= tileRange[1] ) * ( keypoints1[:,1] <= tileRange[3] )         
        tile_descriptors1 =  descriptors1 [bin_descriptors1,:]    
        ids_descriptors1   =  np.where(bin_descriptors1)[0]
        
        bin_descriptors0  = ( keypoints0[:,0] >= ck_tileRange[0] ) * ( keypoints0[:,0] <= ck_tileRange[2] )
        bin_descriptors0 *= ( keypoints0[:,1] >= ck_tileRange[1] ) * ( keypoints0[:,1] <= ck_tileRange[3] ) 
        tile_descriptors0 =  descriptors0 [bin_descriptors0,:]       
        ids_descriptors0  =  np.where(bin_descriptors0)[0]
        
        if len(tile_descriptors0 ) > 0 and len(tile_descriptors1 ) > 0:
            #''' Skimage
            # Match keypoints
            tile_matches01 = match_descriptors(tile_descriptors0, tile_descriptors1, metric ='hamming' , cross_check=True)
            
            # sort the matched points according to the distance
            dist = DistanceMetric.get_metric("hamming")
            dh_all = dist.pairwise(tile_descriptors0,tile_descriptors1)         # N by M distance matrix 
            sort_dh = []
            [sort_dh.append( dh_all[tile_matches01[i,:][0],tile_matches01[i,:][1]] )for i in range( len(tile_matches01))]
            sort_index = np.argsort(sort_dh)
            
            # match_perc = 0.3 #e.g.0.30                                   # extract he first % mathced keypoints e.g.0.3
            numGoodMatches = max(1, int(len(sort_index) * match_perc))
            # print ("numGoodMatches=", numGoodMatches)

            # tile_matches01 = tile_matches01[sort_index]                         # ascending order of dist
            tile_matches01 = tile_matches01[sort_index[:numGoodMatches]]          # the most similar ones
            if len(tile_matches01) > 0 :
                # Select keypoints from
                #   * source (image to be registered)   : pano0
                #   * target (reference image)          : pano1, no move
                # attach the nw keyppoints
                tile_src = keypoints0[ids_descriptors0 [tile_matches01[:, 0]]][:, ::-1] 
                tile_dst = keypoints1[ids_descriptors1 [tile_matches01[:, 1]]][:, ::-1]    

                src   = np.concatenate((src , tile_src ), axis=0)
                dst   = np.concatenate((dst , tile_dst ), axis=0)
            #'''
            

            ''' CV2
            #Matching
            matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
            matches = matcher.match(tile_descriptors0, tile_descriptors1)
            # Sort matches by score
            matches.sort(key=lambda x: x.distance, reverse=False)
            # Remove not so good matches
            match_perc = 0.15
            numGoodMatches = int(len(matches) * match_perc)
            matches = matches[:numGoodMatches]
            # Extract location of good matches
            target_points = np.zeros((len(matches), 2), dtype=np.float32)
            source_points = np.zeros((len(matches), 2), dtype=np.float32)
            for i, match in enumerate(matches):
                target_points[i, :] = target_keypoints[match.queryIdx].pt
                source_points[i, :] = source_keypoints[match.trainIdx].pt

            target_points = np.floor(target_points)
            source_points = np.floor(source_points)
 
            '''
    src  = src[1:, :]
    dst  = dst[1:, :]
    return src,dst

def transfromest_tiled(keypoints0,descriptors0,keypoints1,descriptors1, paras, tilerange_ls ,ck_shift, verbose = False):
    if paras.multiprocess == False:  # use single process option
        src,dst = match_descriptors_tiled (keypoints0, descriptors0,
                                         keypoints1,descriptors1, 
                                         paras.target_shape, tilerange_ls,
                                         ck_shift = paras.ck_shift,
                                         match_perc = paras.match_perc)        
    else:                              # use multiprocess option
        # set numofthreads
        if str(paras.multiprocess).isnumeric():
            numofthreads = int(paras.multiprocess)
        else:
            numofthreads = multiprocessing.cpu_count()
                 
        pool = ThreadPool(processes=numofthreads)
        ls_size = int(np.ceil(len(tilerange_ls)/numofthreads))
        if verbose:
            print ("ls_size =",ls_size)

        # run multiprocessin
        async_result = []
        for th in range (0, numofthreads):
            tilerange_ls_mp   = tilerange_ls[  th*ls_size: th*ls_size +ls_size ]     # split the whole tilerange_ls in to parts for multiprocessing  
            if len(tilerange_ls_mp) > 0 : 
                async_result.append(  
                        pool.apply_async( 
                                match_descriptors_tiled, ( 
                                        keypoints0,descriptors0,keypoints1,descriptors1, 
                                        paras.target_shape, tilerange_ls_mp ,ck_shift,paras.match_perc
                                    )))  # tuple of args for foo
                if verbose:
                    print("\tmulti thread for", th, " ... ,","len(tilerange_ls_mp)=",len(tilerange_ls_mp))
                        
        pool.close()        
        pool.join()
        # load results
        src   = np.zeros((1, 2))
        dst   = np.zeros((1, 2))
        for r in async_result:
            src         = np.concatenate((src    ,   r.get()[0] ), axis=0)
            dst         = np.concatenate((dst    ,   r.get()[1] ), axis=0)

        src  = src[1:, :]
        dst  = dst[1:, :]
        
        print("*" * 10 + " transfromest_tiled: = ", src.shape[0] )

    return src,dst

##############  #''' merge_diff_mask related'''' ####################

def get_miss_mask (binary_diff, min_size = 5000):
    new_centers_mask_bin = morphology.binary_erosion (binary_diff,morphology.disk(6))   
    new_centers_mask_bin = morphology.binary_dilation (new_centers_mask_bin,morphology.disk(60))   
    masks_labels = measure.label ( new_centers_mask_bin)
    for obj in measure.regionprops(masks_labels):
        if obj.area < min_size:
            masks_labels[masks_labels==obj.label] = 0
        else:            
            filled = masks_labels[ obj.bbox[0]:obj.bbox[2],   #i: i+crop_height
                                   obj.bbox[1]:obj.bbox[3]]
            filled = filled*(filled!=obj.label) + obj.filled_image*obj.label    # background +  new filledImage
            masks_labels[ obj.bbox[0]:obj.bbox[2],   #i: i+crop_height
                          obj.bbox[1]:obj.bbox[3]] = filled
#            # fill the bbox rectanglular area with
#            if obj.area /  ( ( obj.bbox[2]-obj.bbox[0] ) * (obj.bbox[3]-obj.bbox[1]) ) > 0.8:
#                masks_labels[ obj.bbox[0]:obj.bbox[2],   #i: i+crop_height
#                              obj.bbox[1]:obj.bbox[3]] = obj.label 

    return masks_labels
        
def get_miss_mask_tile (binary_diff_ls, min_size = 5000):
    masks_labels_ls= []
    for binary_diff in binary_diff_ls:
        masks_labels = get_miss_mask (binary_diff, min_size = min_size)
        masks_labels_ls.append(masks_labels)
    return masks_labels_ls 
       
def exam_diff_mask_tile  (inital_diff, boostrap_tileRange_ls,numofthreads, min_area):
#    print("numofthreads = ", numofthreads)        
    pool = ThreadPool(processes=numofthreads)
    
    # prepare the data
    inital_diff_crop_ls = []
    for i, tileRange in enumerate( boostrap_tileRange_ls): 
        inital_diff_crop = inital_diff[ tileRange[0]:tileRange[2],   #i: i+crop_height
                                        tileRange[1]:tileRange[3]]
        inital_diff_crop_ls.append(inital_diff_crop)    
            
    ls_size = int(np.ceil(len(inital_diff_crop_ls)/numofthreads))
  
    # run multiprocess
    async_result = []
    for th in range (0, numofthreads):
        inital_diff_crop_ls_mp   = inital_diff_crop_ls[  th*ls_size: th*ls_size +ls_size ]     # split the whole tilerange_ls in to parts for multiprocessing  
        if len(inital_diff_crop_ls_mp) > 0 : 
            async_result.append(  
                    pool.apply_async( 
                            get_miss_mask_tile, ( inital_diff_crop_ls_mp,min_area                                    
                                )))  # tuple of args for foo
#            print("\tmulti thread for", th, " ... ,","len(inital_diff_crop_ls_mp)=",len(inital_diff_crop_ls_mp))
                    
    pool.close()        
    pool.join()
    # load results
    masks_labels_ls = []
    diff_mask = np.zeros_like(inital_diff,dtype = np.bool)   # we don;t need label id, just need positve 
    for r in async_result:
        masks_labels_ls += r.get()
    for i, (tileRange,masks_labels) in enumerate( zip(boostrap_tileRange_ls,masks_labels_ls)): 
        diff_mask[ tileRange[0]:tileRange[2],   #i: i+crop_height
                   tileRange[1]:tileRange[3]] = True #masks_labels
    return diff_mask


def merge_diff_mask ( boostrap_tileRange_ls, inital_diff, paras):
    
    diff_mask = np.zeros_like(inital_diff,dtype = np.bool)
    t0 = time.time()
    print ("'''merge_diff_mask '''")
    print (" Step1 : exam all the small tile, fill the masks tile by tile")
    min_area = ( paras.tile_shape[0] *paras.tile_shape[1]/4 ) 
    max_area = ( paras.tile_shape[0] *paras.tile_shape[1]*10 )     
    if len(boostrap_tileRange_ls) > 0 :    
        if paras.multiprocess == False:  # use single process option
            for i, tileRange in enumerate( boostrap_tileRange_ls): 
                inital_diff_crop = inital_diff[ tileRange[0]:tileRange[2],   #i: i+crop_height
                                                tileRange[1]:tileRange[3]]
                    
                diff_mask[ tileRange[0]:tileRange[2],   #i: i+crop_height
                           tileRange[1]:tileRange[3]] = get_miss_mask (inital_diff_crop,
                                                                           min_size = min_area/10)            
        else:                              # use multiprocess option
            # set numofthreads
            if str(paras.multiprocess).isnumeric():
                numofthreads = int(paras.multiprocess)
            else:
                numofthreads = multiprocessing.cpu_count()
            diff_mask = exam_diff_mask_tile  (inital_diff, boostrap_tileRange_ls,numofthreads, min_area/10)
            
    t1=  time.time()
    print ("Used time = ", t1-t0)
    print (" Step2 :  Merge the diff mask result into diff_label_final")
    diff_mask = np.array( diff_mask > 0, dtype = np.int)

    # diff_final = measure.label(diff_mask)      # shows error when imag size is extremly large
    diff_final = ndimage.label(diff_mask)[0]           
    
    current_max_label = diff_final.max()
    
    t2=  time.time()
    print ("Used time = ", t2-t1)
    print ( " Step3 :  check if size too big " ,"current_max_label=",current_max_label )
    for obj in measure.regionprops(diff_final) :
        if  obj.area < min_area*4:
            diff_final[ obj.bbox[0]:obj.bbox[2],   #i: i+crop_height
                        obj.bbox[1]:obj.bbox[3]] = 0            
#        if obj.area > max_area:     #:   # > min: accept ;  break in to two even parts
#            filled_image = obj.filled_image
#            filled_image[ :, 0:int (filled_image.shape[1]/2)] = obj.filled_image[ 
#                          :, 0:int (filled_image.shape[1]/2)] * (current_max_label+1 -obj.label)
#            diff_final[ obj.bbox[0]:obj.bbox[2],   #i: i+crop_height
#                        obj.bbox[1]:obj.bbox[3]] += filled_image                       
#            current_max_label +=1              

    t3 =  time.time()
    print ("Used time = ", t3-t2)              
    print ("diff_label.max() =", diff_final.max())
    return diff_final    
    
    