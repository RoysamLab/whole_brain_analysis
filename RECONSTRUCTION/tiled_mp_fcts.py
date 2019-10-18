# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:29:11 2019

@author: xli63
"""
import numpy as np
from skimage.feature import ORB, match_descriptors
from skimage.transform import ProjectiveTransform, SimilarityTransform, AffineTransform,PolynomialTransform,resize
import multiprocessing
from multiprocessing.pool import ThreadPool
import warnings
warnings.filterwarnings("ignore")

from sklearn.neighbors import DistanceMetric

def featureExtract_tiled(img, paras, tileRange_ls):
   
    if paras.multiprocess == False:  # use single process option
        keypoint_all, descriptor_all = orb_detect_tiled(img, paras, tileRange_ls)

    else:                              # use multiprocess option
        # Set numOfThreads
        if paras.multiprocess.isnumeric():
            print("&&&&&& Use specified number of numOfThreads ")
            numOfThreads = int(paras.multiprocess)
        else:
            numOfThreads = multiprocessing.cpu_count()
            print("&&&&&&& Use defalut number of numOfThreads")
        print("numOfThreads = ", numOfThreads)        
        pool = ThreadPool(processes=numOfThreads)
        ls_size = int(np.ceil(len(tileRange_ls)/numOfThreads))
        print ("ls_size =",ls_size)
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
                print("\tmulti thread for", th, " ... ,","len(tileRange_ls_mp)=",len(tileRange_ls_mp))
        pool.close()        
        pool.join()
        # load results
        keypoint_alltiles   = np.zeros((1,2))
        descriptor_alltiles = np.zeros((1,256))
        for r in async_result:
            keypoint_alltiles   = np.concatenate((keypoint_alltiles,   r.get()[0]), axis=0)
            descriptor_alltiles = np.concatenate((descriptor_alltiles, r.get()[1]), axis=0)

        keypoint_all   =  keypoint_alltiles[1:, :]
        descriptor_all =  descriptor_alltiles[1:, :]
        print("[multiprocessing] featureExtract_tiled: keypoint_alltiles.shape = ", keypoint_alltiles.shape)

    print("*" * 10 + "Detected keypoint numbers: = ", keypoint_all.shape[0] )

    return keypoint_all,descriptor_all

def orb_detect_tiled(img, paras, tileRange_ls,verbose = 0 ):
    ''' featureExtraction on large images with similar texture distribution '''
    # In order to feature extraction over all regions in the image, we will apply feature extraction tiled by tile
    # crop image into tiles and extract keypointss
    keypoint_alltiles   = np.zeros((1, 2))
    descriptor_alltiles = np.zeros((1, 256))    
    if verbose == 1:
        print ("$"*5 + "[Debug point 0]" )
        print ("len(tileRange_ls)=", len(tileRange_ls))

    for t_i, tileRange in enumerate( tileRange_ls):
        crop_img = img[ tileRange[0]:tileRange[2],   #i: i+crop_height
                        tileRange[1]:tileRange[3]]   #j: j + crop_width     
                       
        # only tiles contain enough engergy (obvious contrast of adjacent pixels) have keypoints                
        tile_n_keypoints = int(paras.n_keypoints / paras.tiles_numbers)  # past average distribute

        orb = ORB(n_keypoints=tile_n_keypoints,
                fast_threshold=paras.fast_threshold,
                harris_k=paras.harris_k)
        #  if without this step, orb.detect_and_extract will return error
        orb.detect(crop_img)
        if verbose == 1:
            print ("$"*5 + "[Debug point 2]" )
            print ("\tPre check whether keypoints are detected or not")
            print ("\tlen(orb.scales)= " , len(orb.scales))

        if len(orb.scales) > 0:
            orb.detect_and_extract(crop_img)
            keypoints0 = orb.keypoints  # n by 2 array, n is number of keypoints
            keypoint = keypoints0 + np.tile([tileRange[0], tileRange[1]],
                                            (keypoints0.shape[0], 1))           # add the crop coordinate shift
            descriptor = orb.descriptors                                        # n by 256 array
            
            if verbose == 1:
                print("\t\t" ,tileRange, "orb_detect_tiled : = ", keypoint.shape[0])

            keypoint_alltiles   = np.concatenate((keypoint_alltiles  , keypoint  ), axis=0)
            descriptor_alltiles = np.concatenate((descriptor_alltiles, descriptor), axis=0)
                

    keypoint_all    = keypoint_alltiles     [1:, :]
    descriptor_all  = descriptor_alltiles   [1:, :]

    return keypoint_all, descriptor_all


def match_descriptors_tiled (keypoints0,descriptors0,keypoints1,descriptors1,
                             paras, tileRange_ls ,ck_shift = 30,verbose =0):
    print (" \n''' match_descriptors_tiled ")
    ck_tileRange_ls = []
    src   = np.zeros((1, 2))
    dst   = np.zeros((1, 2))
    inliers = np.zeros((1))
    for t_i, tileRange in enumerate( tileRange_ls):
        
        ck_tileRange=  [ max( 0 , tileRange[0] - ck_shift ), 
                         max( 0 , tileRange[1] - ck_shift ),
                         min( int(paras.target_shape[0] ) -1, tileRange[2] + ck_shift) , 
                         min( int(paras.target_shape[1] ) -1, tileRange[3] + ck_shift) ]         # extract the labels of subpicious window from previous merged results
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
            # Match keypoints
            tile_matches01 = match_descriptors(tile_descriptors0, tile_descriptors1, metric ='hamming' , cross_check=True)
            # sort the matched points according to the distance
            dist = DistanceMetric.get_metric("hamming")
            dh_all = dist.pairwise(tile_descriptors0,tile_descriptors1)         # N by M distance matrix 
            sort_dh = []
            [sort_dh.append( dh_all[tile_matches01[i,:][0],tile_matches01[i,:][1]] )for i in range( len(tile_matches01))]
            sort_index = np.argsort(sort_dh)
            tile_matches01 = tile_matches01[sort_index]                         # ascending order of dist
                
            if len(tile_matches01) > 1 :
#                tile_matches01 = tile_matches01[:paras.min_samples]          # only select the cloest 10* pairs for each tile 
                # Select keypoints from
                #   * source (image to be registered)   : pano0
                #   * target (reference image)          : pano1, no move
                # attach the nw keyppoints
                tile_src = keypoints0[ids_descriptors0 [tile_matches01[:, 0]]][:, ::-1] 
                tile_dst = keypoints1[ids_descriptors1 [tile_matches01[:, 1]]][:, ::-1]    
                src   = np.concatenate((src , tile_src ), axis=0)
                dst   = np.concatenate((dst , tile_dst ), axis=0)
    
    src  = src[1:, :]
    dst  = dst[1:, :]
    inliers = inliers[1:]
          
    return src,dst


def transfromest_tiled(keypoints0,descriptors0,keypoints1,descriptors1, paras, tilerange_ls ,ck_shift):
    print (" \n''' 2. transform estimation and ransac''' ")
#    paras.multiprocess = False
    if paras.multiprocess == False:  # use single process option
        src,dst = match_descriptors_tiled (keypoints0, descriptors0,
                                         keypoints1,descriptors1, 
                                         paras, tilerange_ls,
                                         ck_shift = paras.ck_shift)        
    else:                              # use multiprocess option
        # set numofthreads
        if paras.multiprocess.isnumeric():
            print("&&&&&& use specified number of numofthreads ")
            numofthreads = int(paras.multiprocess)
        else:
            numofthreads = multiprocessing.cpu_count()
            print("&&&&&&& use defalut number of numofthreads")
            
        print("numofthreads = ", numofthreads)        
        pool = ThreadPool(processes=numofthreads)
        ls_size = int(np.ceil(len(tilerange_ls)/numofthreads))
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
                                        paras, tilerange_ls_mp ,ck_shift
                                    )))  # tuple of args for foo
                print("\tmulti thread for", th, " ... ,","len(tilerange_ls_mp)=",len(tilerange_ls_mp))
                        
        pool.close()        
        pool.join()
        # load results
        model_robust01_inverse_ls = []
        src   = np.zeros((1, 2))
        dst   = np.zeros((1, 2))
        inliers = np.zeros((1))
        for r in async_result:
            src         = np.concatenate((src    ,   r.get()[0] ), axis=0)
            dst         = np.concatenate((dst    ,   r.get()[1] ), axis=0)

        print("\n[multiprocessing] featureextract_tiled: inliers.shape = ", inliers.shape)
        print("\n[multiprocessing] featureextract_tiled: model_robust01_inverse_ls.shape = ", len(model_robust01_inverse_ls))

        src  = src[1:, :]
        dst  = dst[1:, :]
        
        print("*" * 10 + " transfromest_tiled: = ", inliers.shape[0] )

    return src,dst