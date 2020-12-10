# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:55:36 2019

@author: xli63
"""
import numpy as np
from skimage import exposure
#from s import ndimage
from skimage.filters import threshold_otsu
from skimage.util import img_as_uint ,img_as_ubyte
import warnings
from skimage import segmentation,measure,morphology   
warnings.filterwarnings("ignore")
import time

def adjust_image (img):
    p2, p98 = np.percentile(img, (0.02, 99.8))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))          # Load over images
    return img

#%%
def check_shape(img,shape):
    if shape[0] <= img.shape[0] and shape[1] <= img.shape[1]:                    # crop the image
        checked_img =  img [:shape[0] ,:shape[1] ]
    else:                                                                      # zeropadding
        canvas = np.zeros([max(shape[0],img.shape[0]) ,
                           max(shape[1],img.shape[1]) ], dtype = img.dtype )
        canvas[:min( shape[0],img.shape[0] ) ,
               :min( shape[1],img.shape[1] ) ]= img[:min( shape[0],img.shape[0] ) ,
                                                    :min( shape[1],img.shape[1] ) ]
        checked_img = canvas[:shape[0] ,:shape[1] ]
    return checked_img
#%%
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


def eval_draw_diff (target, source, tileRange_ls= None, verbose = False) :   
    
    vis_target = adjust_image( img_as_ubyte( target) )
    vis_source = adjust_image( img_as_ubyte( source))
    if verbose == True:
        print ("\n--------------eval_draw_diff-------------------")
        print ("vis_target:",vis_target.max(),"vis_source:",vis_source.max())

    if vis_source.max() == 0:    # make sure the wrapped is not empty
        return None, None,None,np.inf,None
        
    vis = np.dstack ( [ vis_target,             # RGB 
                        vis_source,
                        np.zeros_like(target,dtype= np.uint8)])     
    # import pdb;pdb.set_trace()
    '''
    Fast binarization, randomly selection some region, take the medium
     of the thresholds
    '''
    crop_weights, crop_heights = [50,50]
    coord_x_ls = np.random.random_integers( 0 , vis_target.shape[0] - 100 , 50 )
    coord_y_ls = np.random.random_integers( 0 , vis_target.shape[1] - 100 , 50 )   # 50 cropps

    '''Fast otsu, randomly select 50 crops and take the median of the cropped thres'''
    thres_source_target = []
    for vis_img in [vis_source, vis_target] : 
        thres_ls = []
        for coord_x,coord_y in zip(coord_x_ls, coord_y_ls ):
            vis_crop = vis_img[ coord_x:coord_x+crop_weights,
                                coord_y:coord_y+crop_heights]
            # import pdb; pdb.set_trace()
            if len( np.unique( vis_crop) )> 1:
                thres_ls.append( threshold_otsu(vis_crop) )
        thres_source_target.append( np.median(thres_ls) )
    #
    binary_source = vis_source >= thres_source_target[0] * 1.2
    binary_target = vis_target >= thres_source_target[1] * 1.2
    binary_diff = abs( binary_source *1- binary_target *1 ) 
    error = binary_diff.sum()/binary_target.sum()* 100
    
    error_map = None
    if tileRange_ls is not None:
        error_map = np.zeros(target.shape, dtype=np.int8)
        for tileRange in  tileRange_ls:
            binary_target_tile = binary_target [ tileRange[0]:tileRange[2],   #i: i+crop_height
                                                 tileRange[1]:tileRange[3]] 
            binary_diff_tile   = binary_diff [ tileRange[0]:tileRange[2],   #i: i+crop_height
                                               tileRange[1]:tileRange[3]] 
            if binary_target_tile.sum() > 0:
                error_map [ tileRange[0]:tileRange[2],   #i: i+crop_height
                            tileRange[1]:tileRange[3]] = int ( binary_diff_tile.sum()/binary_target_tile.sum()* 100)
            
    return vis, binary_diff,binary_target,error,error_map


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
        if type(misalignRange_ls) == list : 
            for tileRange in  misalignRange_ls:
                vis_diff [ tileRange[0]:tileRange[2]  , tileRange[1]:tileRange[1]+3,: ] = [0,255,255]    #i: i+crop_height
                vis_diff [ tileRange[0]:tileRange[0]+3, tileRange[1]:tileRange[3]  ,: ] = [0,255,255]    #i: i+crop_height
                vis_diff [ tileRange[0]:tileRange[2]  , tileRange[3]:tileRange[3]+3,: ] = [0,255,255]    #i: i+crop_height
                vis_diff [ tileRange[2]:tileRange[2]+3, tileRange[1]:tileRange[3]  ,: ] = [0,255,255]    #i: i+crop_height   
        else:
            border = segmentation.find_boundaries(misalignRange_ls)
            vis_diff[border >0] = [0,255,255]
                            
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

def merge_tileRange(tileRange_ls,max_size = [8000,8000], min_size = [1000,1000]) :
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
                        tileRange_merged_temp=  [ max(0,  min (tileRange[0], exam_tileRange[0] - min_size[0])),
                                                  max(0,  min (tileRange[1], exam_tileRange[1] - min_size[0])),
                                                  max (tileRange[2], exam_tileRange[2] + min_size[0]),
                                                  max (tileRange[3], exam_tileRange[3] + min_size[1]),
                                                     ]
                        visited_tileRange_ls.append(exam_tileRange)
                        tileRange_merged = tileRange_merged_temp

            if tileRange_merged == None: 
                tileRange_merged_ls.append(tileRange)   
            else:
                tileRange_merged_ls.append(tileRange_merged)   
    return tileRange_merged_ls

