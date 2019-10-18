# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:55:36 2019

@author: xli63
"""
import numpy as np
from skimage import exposure
from skimage.filters import threshold_otsu
from skimage.util import img_as_uint ,img_as_ubyte,img_as_float
import warnings
warnings.filterwarnings("ignore")

def adjust_image (img):
    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))          # Load over images
    return img

def check_shape(img,shape):
    if img.shape[0] > shape[0] and img.shape[1] > shape[1] :
        checked_img = img [:shape[0],:shape[1]]
    else:
        checked_img = np.zeros((shape[0], shape[1]), dtype = img.dtype )
        checked_img [:img.shape[0] ,:img.shape[1] ] = img
    return checked_img

def crop_tiles( img_shape,tile_shape,crop_overlap = 0,ck_shift=None):
    (img_rows,img_cols) =  img_shape  
    (tile_height,tile_width)  = tile_shape     
    if ck_shift == None:
        tileRange_ls = []   
        for i in range(0, img_rows, tile_height- crop_overlap ):
           for j in range(0, img_cols, tile_width -crop_overlap):                   
               tileRange = [ i,j,
                              min(img_rows, tile_height + i),
                              min(img_cols, tile_width  + j )]
               tileRange_ls.append ( tileRange )
        result = tileRange_ls
    else:
        ck_tileRange_ls = []
        for i in range(0, img_rows, tile_height - crop_overlap ):
           for j in range(0, img_cols, tile_width - crop_overlap):
               ck_tileRange=  [  max( 0 , i - ck_shift ), 
                                 max( 0 , j - ck_shift ),
                                 min( img_rows, tile_height + i + ck_shift) , 
                                 min( img_cols, tile_width  + j + ck_shift) ]         # extract the labels of subpicious window from previous merged results
               ck_tileRange_ls.append(ck_tileRange)    
        result = ck_tileRange_ls  
    return result


def eval_draw_diff (target, source, tileRange_ls= None, err_thres = 1) :   
#    import pdb ; pdb.set_trace()  
    print ("\n--------------eval_draw_diff-------------------")
    vis_target = adjust_image( img_as_ubyte( target) )
    vis_source = adjust_image( img_as_ubyte( source))
    print ("vis_target:",vis_target.max(),"vis_source:",vis_source.max())
    vis = np.dstack ( [ vis_target,             # RGB 
                        vis_source,
                        np.zeros_like(target,dtype= np.uint8)])     
            
    binary_source = vis_source >= threshold_otsu(vis_source) * 1.2
    binary_target = vis_target >= threshold_otsu(vis_target) * 1.2
    binary_diff = abs( binary_source *1- binary_target *binary_target *1 ) 
    error = binary_diff.sum()/binary_target.sum()* 100

    misalignRange_ls = []
    if tileRange_ls is not None : 
        for tileRange in tileRange_ls:
            crop_binary_diff = binary_diff[ tileRange[0]:tileRange[2],   #i: i+crop_height
                                            tileRange[1]:tileRange[3]]   #j: j + crop_width    
            crop_binary_target = binary_target[ tileRange[0]:tileRange[2],   #i: i+crop_height
                                                tileRange[1]:tileRange[3]]   #j: j + crop_width    
    
            crop_error = crop_binary_diff.sum()/crop_binary_target.sum()
#            print ("all :mean",crop_binary_target.mean(), "error", crop_error) # (0.008 ~0.1;0.2~1.5)
            if  ( crop_error > err_thres and                                    # make sure the misalign area is in large porportion
                  crop_binary_target.mean() > 0.05 ):                           # make sure the intensity of raw image is high enough
                misalignRange_ls.append(tileRange)
#                print ("tileRange:",tileRange,"crop_error =", crop_error)

    return vis, binary_diff,error,misalignRange_ls


def differenceVis (binary_diff,kps,inliers=None, tileRange_ls = None,misalignRange_ls = None):
    vis_diff = np.dstack ( [ img_as_ubyte(binary_diff*255),             # RGB 
                            np.zeros_like(binary_diff,dtype= np.uint8),
                            np.zeros_like(binary_diff,dtype= np.uint8)])    
            
    for i in range(kps.shape[0]):
        x = int(kps[i,1])
        y = int(kps[i,0])
        vis_diff[ max( 0, x -3)  : min( binary_diff.shape[0], x + 3 ), 
                max( 0, y -3)  : min( binary_diff.shape[1], y + 3 ), : ]= [255,255,255]      # white
        if inliers is not None:
            if inliers[i] == True : 
                vis_diff[ max( 0, x -3)  : min( binary_diff.shape[0], x + 3 ), 
                        max( 0, y -3)  : min( binary_diff.shape[1], y + 3 ), :]= [0,255,0]     # G                   
    # drow the grid
    if tileRange_ls is not None:
        for tileRange in  tileRange_ls:
            vis_diff [ tileRange[0]:tileRange[2]  , tileRange[1]:tileRange[1]+3,: ] = 255   #i: i+crop_height
            vis_diff [ tileRange[0]:tileRange[0]+3, tileRange[1]:tileRange[3]  ,: ] = 255   #i: i+crop_height
            vis_diff [ tileRange[0]:tileRange[2]  , tileRange[3]:tileRange[3]+3,: ] = 255   #i: i+crop_height
            vis_diff [ tileRange[2]:tileRange[2]+3, tileRange[1]:tileRange[3]  ,: ] = 255   #i: i+crop_height            
    if misalignRange_ls is not None:
        for tileRange in  misalignRange_ls:
            vis_diff [ tileRange[0]:tileRange[2]  , tileRange[1]:tileRange[1]+3,: ] = [0,255,255]    #i: i+crop_height
            vis_diff [ tileRange[0]:tileRange[0]+3, tileRange[1]:tileRange[3]  ,: ] = [0,255,255]    #i: i+crop_height
            vis_diff [ tileRange[0]:tileRange[2]  , tileRange[3]:tileRange[3]+3,: ] = [0,255,255]    #i: i+crop_height
            vis_diff [ tileRange[2]:tileRange[2]+3, tileRange[1]:tileRange[3]  ,: ] = [0,255,255]    #i: i+crop_height           
                
    return vis_diff

def spl_tiled_data (data, tileRange_ls):
    spl_tile_ls = []
    spl_tile_dic ={}
    target_shape = max(tileRange_ls ) [2:]
    keypoints0,keypoints1 = data
    for t_i, tileRange in enumerate( tileRange_ls):
        ck_shift = 50        
        ck_tileRange=  [ max( 0 , tileRange[0] - ck_shift ), 
                         max( 0 , tileRange[1] - ck_shift ),
                         min( int(target_shape[0] ) -1, tileRange[2] + ck_shift) , 
                         min( int(target_shape[1] ) -1, tileRange[3] + ck_shift) ]         # extract the labels of subpicious window from previous merged results
        bin_kp  = ( keypoints1[:,0] >= tileRange[0] ) * ( keypoints1[:,0] <= tileRange[2] )
        bin_kp *= ( keypoints1[:,1] >= tileRange[1] ) * ( keypoints1[:,1] <= tileRange[3] )         
        bin_kp  = ( keypoints0[:,1] >= ck_tileRange[0] ) * ( keypoints0[:,1] <= ck_tileRange[2] )
        bin_kp *= ( keypoints0[:,0] >= ck_tileRange[1] ) * ( keypoints0[:,0] <= ck_tileRange[3] ) 
        ids_bin_kep  =  np.where(bin_kp)[0]        
        spl_tile_dic[t_i] = ids_bin_kep
        if len(ids_bin_kep) > 5:
            spl_tile_ls.append(ids_bin_kep)    
    return  spl_tile_ls,spl_tile_dic

def merge_tileRange(tileRange_ls,max_size = [8000,8000]) :
    tileRange_merged_ls = []
    visited_tileRange_ls = []
    for tileRange  in tileRange_ls:
        if tileRange not in visited_tileRange_ls:
            tileRange_merged = None
            for exam_tileRange in tileRange_ls:
                if (exam_tileRange != tileRange 
                   and  exam_tileRange not in visited_tileRange_ls ):
                    if     ( tileRange[0] <= exam_tileRange[0]
                         and tileRange[1] <= exam_tileRange[1]
                         and tileRange[2] >= exam_tileRange[0]
                         and tileRange[3] >= exam_tileRange[1]) : 
                        tileRange_merged_temp=  [ min (tileRange[0], exam_tileRange[0]),
                                                       min (tileRange[1], exam_tileRange[1]),
                                                       max (tileRange[2], exam_tileRange[2]),
                                                       max (tileRange[3], exam_tileRange[3]),
                                                     ]
                        visited_tileRange_ls.append(exam_tileRange)
                        tileRange_merged_temp_shape = [ tileRange_merged_temp[2] - tileRange_merged_temp[0],
                                                        tileRange_merged_temp[3] - tileRange_merged_temp[1] ] 
                        if ( tileRange_merged_temp_shape[0] < max_size[0]  and  
                             tileRange_merged_temp_shape[1] < max_size[1] ) :     # make sure the merged size is not too big 
                            tileRange_merged = tileRange_merged_temp

            if tileRange_merged == None: 
                tileRange_merged_ls.append(tileRange)   
            else:
                tileRange_merged_ls.append(tileRange_merged)   
    return tileRange_merged_ls