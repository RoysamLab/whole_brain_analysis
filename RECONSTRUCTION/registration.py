'''
Image Registration of Channels for multi-rounds

Author: Rebecca LI, University of Houston, Farsight Lab, 2019
xiaoyang.rebecca.li@gmail.com

--- inspired  from ---
# https://github.com/scikit-image/skimage-tutorials/blob/master/lectures/adv3_panorama-stitching.ipynb

Improvements compared to above 
1) Feasibility to run on large image 
2) Dramatically improve alignment accuracy for texture based images by tile-based keypoint detection
3) Support multiprocess acceleration & single thread by tile-based keypoint extracting and matcching
4) Auto-evaluation of the result and self-correction for artificall error (local fold)
5) 11/18/2020 (With Jahandar) Use CV2  rather than skimage for keypoint extraction to acclerate 
6) 11/19/2020 (With Jahandar) Try 3 differnet transformation and use the one has the mx inliner_rate

--- conda packages ---
conda install -c conda-forge scikit-image \
conda install scikit-learn \

------ e.g. ------
python registration.py \
    -i [inputPath] \
    -o [outputPath]

---- debugging -----
python registration.py \
    -i [inputPath] \
    -o [outputPath] \
    --demo True
# will generate keypoints, descriptors npy files and intermediate results, and resue it by: (will save time )

python registration.py \
    -i [inputPath] \
    -o [outputPath] \
    --keypoint_dir [outputPath] \
    --demo True

'''

import os
import re
import sys
import glob
import time
import numpy as np
import shutil
import skimage,cv2
from skimage.color import rgb2gray
from skimage.feature import ORB, match_descriptors
from skimage import exposure,io,segmentation,img_as_ubyte,img_as_uint,measure,morphology   
from skimage.transform import warp, ProjectiveTransform, SimilarityTransform, AffineTransform,PolynomialTransform,resize
from skimage.filters import threshold_otsu

import matplotlib.pyplot as plt
from skimage.external import tifffile as tiff
from multiprocessing.pool import ThreadPool
import warnings
warnings.filterwarnings("ignore")
import argparse
from sklearn.neighbors import DistanceMetric

import tiled_mp_fcts as tiled_fcts
import vis_fcts as vis_fcts
from ransac_tile import ransac_tile

class Paras(object):
    def __init__(self):

        # Parameters for tiling
        self.target_shape = (4000,8000)
        self.tile_shape = (1000, 1000)         # # the smaller the tilling, longer the time, better the results
        self.multiprocess = False
        self.crop_overlap = 10  

        # Parameters for ORB
        self.n_keypoints = 300000
        self.fast_threshold = 0.08
        self.harris_k = 0.1


        # Parameters for keypoint matching
        self.match_perc= 1.0 # percentarge of best match keypoints to keep in each tile  (1-0)
        self.keypoint_dir = None
        self.ck_shift = 100  # check window dilation width for seaching robust keypoint   # 50
        self.keypoint_tool= "cv2"           # cv2 or skimage

        # Parameters for ransac model estrimation
        self.min_samples = 5
        self.residual_threshold = 5  # default =1, 7
        self.max_trials = 1000
        self.random_state = 42 #230
        self.ransac_obj= "inlier_num"           #[ "inlier_num","inlier_tile", "residuals_sum", "fast_diff"]
        self.ds_rate= 20                      #ransac_downscale rate: int > 1

        # other user defined paras:
        self.demo = False
        self.bootstrap = True
        self.imadjust = True

        # self.pre_register = False

    def display(self):
        print("============================")
        print("\nParameters for tiling : ")
        print("\ttile_shape = ", self.tile_shape)
        print("\ttarget_shape = ", self.target_shape)
        print("\tck_shift = ", self.ck_shift)
        print("\tcrop_overlap = ", self.crop_overlap)

        print("\nParameters for ORB : ")
        print("\tn_keypoints = ", self.n_keypoints)
        print("\tfast_threshold = ", self.fast_threshold)
        print("\tharris_k = ", self.harris_k)
        print("\tkeypoint_dir = ", self.keypoint_dir)
        print("\tkeypoint_tool = ", self.keypoint_tool)

        print("\nParameters for ransac : ")
        print("\tmin_samples = ", self.min_samples)
        print("\tresidual_threshold = ", self.residual_threshold)
        print("\tmax_trials = ", self.max_trials)
        print("\trandom_state = ", self.random_state)
        print("\tds_rate = ", self.ds_rate)


        print("\nParameters for matching : ")
        print("\tmatch_perc = ", self.match_perc)

        print("\nother nParameters: ")
        print("\tmultiprocess = ", self.multiprocess)
        print("\tdemo = ", self.demo)
        print("\tbootstrap = ", self.bootstrap)
        print("\timadjust = ", self.imadjust)
        print("\transac_obj = ", self.ransac_obj)

        print("============================")


def spl_tiled_data (data, tileRange_ls ,paras):
    spl_tile_ls = []
    spl_tile_dic ={}
    target_shape = max(tileRange_ls ) [2:]
    keypoints0,keypoints1 = data
    for t_i, tileRange in enumerate( tileRange_ls):
        ck_tileRange=  [ max( 0 , tileRange[0] - paras.ck_shift ), 
                         max( 0 , tileRange[1] - paras.ck_shift ),
                         min( int(target_shape[0] ) -1, tileRange[2] + paras.ck_shift) , 
                         min( int(target_shape[1] ) -1, tileRange[3] + paras.ck_shift) ]         # extract the labels of subpicious window from previous merged results
        bin_kp  = ( keypoints1[:,0] >= tileRange[0] ) * ( keypoints1[:,0] <= tileRange[2] )
        bin_kp *= ( keypoints1[:,1] >= tileRange[1] ) * ( keypoints1[:,1] <= tileRange[3] )         
        bin_kp  = ( keypoints0[:,1] >= ck_tileRange[0] ) * ( keypoints0[:,1] <= ck_tileRange[2] )
        bin_kp *= ( keypoints0[:,0] >= ck_tileRange[1] ) * ( keypoints0[:,0] <= ck_tileRange[3] ) 
        ids_bin_kep  =  np.where(bin_kp)[0]        
        spl_tile_dic[t_i] = ids_bin_kep
        if len(ids_bin_kep) > paras.min_samples /2:           # number of keypoints in this tile have to be bigger than 
            spl_tile_ls.append(ids_bin_kep)    
    return  spl_tile_ls,spl_tile_dic

def local_register(target0,source0,paras):
    orb = ORB(n_keypoints= paras.n_keypoints, fast_threshold=paras.fast_threshold)
    # Detect keypoints in pano0
    orb.detect_and_extract(source0)
    keypoints0 = orb.keypoints
    descriptors0 = orb.descriptors

    orb.detect_and_extract(target0)
    keypoints1 = orb.keypoints
    descriptors1 = orb.descriptors

    matches01 = match_descriptors(descriptors0, descriptors1, cross_check=True)
    src = keypoints0[matches01[:, 0]][:, ::-1]
    dst = keypoints1[matches01[:, 1]][:, ::-1]
    model_robust01, inliers01 = ransac((src, dst), AffineTransform,
                                                    min_samples         = paras.min_samples, 
                                                    residual_threshold  = paras.residual_threshold,
                                                    max_trials          = paras.max_trials,
                                                    random_state        = paras.random_state,
                                )      
    print ("Local register:detect keypoints",descriptors0.shape[0], "matched",sum(matches01))
    return model_robust01
    
def select_keypointsInMask(kp, binmask):
    # kp = N by 2
    # binmask is same size as image, 1
    selected_kp = np.zeros(kp.shape[0], dtype = np.bool)
    for i, (x, y) in enumerate( zip( kp[:,0],kp[:,1]) ) :
        if binmask[int(x),int(y)] > 0:
            selected_kp[i] = True                 
    return selected_kp

def fast_error_estimate(model,source, target):    
    '''
    error = binary_diff.sum()/binary_target.sum()* 100   # smaller the better
    '''
    source_warped = warp(source, inverse_map = model.inverse, 
                                 output_shape = target.shape, cval=0)             # float64

    # source_warped = warp(source0_ds, inverse_map = model_tf.inverse,  output_shape = target0_ds.shape, cval=0)             # float64
    if source_warped.max() == 0 :
        return np.inf
    else:
        __, __,__,error,__ = vis_fcts.eval_draw_diff ( target, source_warped )                   

        return error

def registrationORB_tiled(targets, sources, paras, output_dir, 
                          save_target=True, 
                          keypoints1=None, descriptors1=None,  verbose =0):    
    t_0 = time.time()    
    target0 = []  # only use target 0 and source 0 for keypoint extraction
    source0 = []

    # READING IMAGES
    for key_id, t_key in enumerate( sorted ( targets.keys() )) :                
        if key_id == 0:
            with tiff.TiffFile(targets[t_key]) as tif:
                target0 = tif.asarray(memmap=True)
            input_type = target0.dtype
            target0 = rgb2gray(target0) if target0.ndim == 3 else target0
            # print("Process ", t_key, " as target0", "target0.shape = ", target0.shape)
            target0_key = t_key
    for key_id, s_key in enumerate( sorted ( sources.keys() )) :
        if key_id == 0:
            with tiff.TiffFile(sources[s_key]) as tif:
                source0 = tif.asarray(memmap=True)            
            source0 = rgb2gray(source0) if source0.ndim == 3 else source0
            # print("Process ", s_key, " as source0", "source0.shape =", source0.shape)
            source0_key = s_key

    if target0 is [] or source0 is []:
        print("!! target or source not found error:", sys.exc_info()[0])

    '''1. Feature detection and matching'''
    # convert to 8 bit for detection in loweer quality 
    target0 = img_as_ubyte(target0) if input_type != np.uint8  else target0
    source0 = img_as_ubyte(source0) if input_type != np.uint8  else source0

    target0 = vis_fcts.adjust_image (target0) if paras.imadjust is True  else target0
    source0 = vis_fcts.adjust_image (source0) if paras.imadjust is True else source0
    assert target0.max()!=0    # make sure the wrapped is not empty
    assert source0.max()!=0    # make sure the wrapped is not empty

    paras.target_shape = ( target0.shape[0],target0.shape[1] )
    
    paras.display()
    target0_mean = target0.mean()
    source0 = vis_fcts.check_shape(source0,paras.target_shape)                        

    t_8bit = time.time()
    
    # set tile ranges list 
    if paras.tile_shape == []:  # do not apply any tilling
        tile_shape = (int(target0.shape[0]), int(target0.shape[1]))
    else:  # default tile shape is 400*800
        tile_shape = paras.tile_shape
    tile_width, tile_height    = tile_shape
    img_rows = int(target0.shape[0])
    img_cols = int(target0.shape[1])                    

    tileRange_ls    = vis_fcts.crop_tiles(  ( img_rows,img_cols ),
                                           ( tile_width, tile_width),
                                           paras.crop_overlap)  
    paras.tiles_numbers = len(tileRange_ls)

    if verbose == 1 : 
        print ("tile_shape=" ,tile_shape)
        print("number of the tiles = ", paras.tiles_numbers)
        print("tileRange_ls[0] = ", tileRange_ls[0] )

    print(''' 1.1.  EXTRACT KEYPOINTS ''')
    if paras.keypoint_dir == None:
        keypoints0, descriptors0 = tiled_fcts.featureExtract_tiled(source0, paras, tileRange_ls)  # keypoints0.max(axis=1)
        if keypoints1 is None or descriptors1 is None:  # need to create featureExtraction for target, else read the created one from input
            keypoints1, descriptors1 = tiled_fcts.featureExtract_tiled(target0, paras, tileRange_ls)
        
        if paras.demo ==True:
            np.save( os.path.join ( output_dir, source0_key.split(".")[0] + "_keypoints0.npy"),keypoints0)
            np.save( os.path.join ( output_dir, source0_key.split(".")[0] + "_descriptors0.npy"),descriptors0)
            np.save( os.path.join ( output_dir, target0_key.split(".")[0] + "_keypoints1.npy"),keypoints1)
            np.save( os.path.join ( output_dir, target0_key.split(".")[0] + "_descriptors1.npy"),descriptors1)   
            print ("EXTRACT KEYPOINTS have been saved")
    else:
        keypoints0      = np.load( os.path.join ( paras.keypoint_dir, source0_key.split(".")[0] + "_keypoints0.npy"))
        descriptors0    = np.load( os.path.join ( paras.keypoint_dir, source0_key.split(".")[0] + "_descriptors0.npy"))
        keypoints1      = np.load( os.path.join ( paras.keypoint_dir, target0_key.split(".")[0] + "_keypoints1.npy"))
        descriptors1    = np.load( os.path.join ( paras.keypoint_dir, target0_key.split(".")[0] + "_descriptors1.npy"))    
        print ("keypoints0.shape=", keypoints0.shape, "keypoints1.shape = ",keypoints1.shape)
        print ("EXTRACT KEYPOINTS have been loaded from path")

    t_featureExtract_tiled = time.time()
    print("[Timer] featureExtract tiled used time (h) =", str((t_featureExtract_tiled - t_8bit) / 3600))
    
    print("\n",''' 1.2 Match descriptors between target and source image ''')
    # Reorderded keypoints acoording to the matching streateger by descriptors
    if paras.multiprocess == False:
        src,dst =  tiled_fcts.match_descriptors_tiled(keypoints0,descriptors0,
                                          keypoints1,descriptors1, 
                                          paras.target_shape, tileRange_ls ,
                                          ck_shift = paras.ck_shift)

    else:
        src,dst =  tiled_fcts.transfromest_tiled (keypoints0,descriptors0,
                                          keypoints1,descriptors1, 
                                          paras, tileRange_ls ,
                                          ck_shift = paras.ck_shift)


    print ("num of matched desciptors =", len(src[:,0]))
    t_matchKyepoint = time.time()

    print("[Timer] Match descriptors used time (h) =", str((t_matchKyepoint-t_featureExtract_tiled) / 3600))


    '''  LMEDS: Least-Median robust method'''
    # model_robust01, __ = cv2.findHomography(src, dst,  cv2.LMEDS, 5.0)

    print("\n",''' 2. Transform estimation ''' )
    exam_tileRange_ls    = vis_fcts.crop_tiles(  ( img_rows,img_cols ),
                                                ( int( tile_height/2), int( tile_width/2)))  
    spl_tile_ls, spl_tile_dic = spl_tiled_data ( (src, dst) , exam_tileRange_ls, paras)

    print ("Ransac_tile: estimate the transformation" )   # try out all the transforma until the non transoformation has found
    estimated_error_min = np.inf

    '''run for a few iterations for different parameters'''
    # define the downscales size for fast evaluation
    ds_rate = 1
    source0_ds = cv2.resize(source0, (source0.shape[0] // ds_rate, source0.shape[1] // ds_rate))  
    target0_ds = cv2.resize(target0, (target0.shape[0] // ds_rate, target0.shape[1] // ds_rate))  

    #paras.ransac_obj= "fast_diff"
    for tform in [ AffineTransform,ProjectiveTransform]:  #, SimilarityTransform]:        
        if paras.ransac_obj == "fast_diff":
            model_tf, inliers_tf = ransac_tile((src, dst), tform,
                                    max_trials          = paras.max_trials,
                                    min_samples         = paras.min_samples,                #  not matter for "fast_diff"
                                    residual_threshold  = paras.residual_threshold,
                                    random_state        = paras.random_state,
                                    spl_tile_ls         = spl_tile_ls,
                                    verbose             = False, #paras.demo, 
                                    obj_type            = "fast_diff",
                                    source_target       = [source0_ds,target0_ds]
                                    )  
            estimated_error = fast_error_estimate(model_tf,source0_ds, target0_ds)

            if np.all(np.isnan(model_tf.params)) == False :               # only save the model if not Nan
                if estimated_error < estimated_error_min:
                    model_robust01 = model_tf
                    inliers = inliers_tf
                    estimated_error_min=  estimated_error           # update the min error
                    print ( "\t[Valid]estimated_error%  =" , '%.2f'%estimated_error )  
        else:
            for min_samples in range(3,7):
                for residual_threshold in range(1,10):
                    model_tf, inliers_tf = ransac_tile((src, dst), tform,
                                            max_trials          = paras.max_trials,
                                            min_samples         = min_samples,               
                                            residual_threshold  = residual_threshold,
                                            random_state        = paras.random_state,
                                            spl_tile_ls         = spl_tile_ls,
                                            verbose             = False, #paras.demo,
                                            obj_type            = paras.ransac_obj
                                           )  
                    estimated_error = fast_error_estimate(model_tf,source0_ds, target0_ds)

                    # import pdb;pdb.set_trace()

                    if np.all(np.isnan(model_tf.params)) == False :               # only save the model if not Nan
                        if estimated_error < estimated_error_min:
                            model_robust01 = model_tf
                            inliers = inliers_tf
                            estimated_error_min=  estimated_error           # update the min error
                            print ( "\t[Valid]estimated_error%  =" , '%.2f'%estimated_error )                      

    assert model_robust01 is not None                                   # if source wraped ==0, then 9 
    assert np.all(np.isnan(model_robust01.params)) == False            # make sure the model is not nan  

    print ("\n Final testimated_error%  =" , '%.2f'%estimated_error_min )
    t_transformEstimation = time.time()
    print("[Timer] Transform used time (h) =", str((t_transformEstimation-t_matchKyepoint) / 3600))

    print("\n",''' 3. Image Warping''')
    # we must warp, or transform, two of the three images so they will properly align with the stationary image.    
    # Apply same offset on all rest images       
    model_mini_dic = {}
    # save_vis = True if paras.demo == True else False
    reuse_kp = True    # whether to reused keypoint for bootstrap region (not recommend, fast but poor align)
    bootstrap = paras.bootstrap    # bootstrap on potential folded regions 
    
    if paras.demo == False:
        del src,dst,source0
        
    if bootstrap == False:
        del keypoints0, descriptors0

    for s_i, source_key in enumerate( sorted ( sources.keys() )) :
        with tiff.TiffFile(sources[source_key]) as tif:
            source = tif.asarray(memmap=True)              
        source = rgb2gray(source) if source.ndim == 3 else source               # uint16
        
        source = vis_fcts.check_shape(source,paras.target_shape)                # make sure the source image is the same size to target      

        # source_warped = cv2.warpPerspective(source, model_robust01, paras.target_shape)        # LMEDS

        source_warped = warp(source, inverse_map = model_robust01.inverse, 
                                     output_shape = paras.target_shape, cval=0)             # float64

        print('''evaluate the initial registration result ''')
        if (s_i == 0 and bootstrap==True) or paras.demo == True:                                      
            # merge diff regions
            __, inital_diff,binary_target,error,__ = vis_fcts.eval_draw_diff ( img_as_ubyte(target0),                                                            
                                                                               img_as_ubyte(source_warped) )                   
            bsBoost_tileRange_ls = []
            for tileRange_key in spl_tile_dic.keys() :
                tileRange = exam_tileRange_ls[tileRange_key]            #exam tile range (small)
                spl_tile =  spl_tile_dic[tileRange_key]
                inlier_tile = inliers [ spl_tile]
                inlier_rate = inlier_tile.sum() / len(inlier_tile)                          
                crop_target0 = target0[ tileRange[0]:tileRange[2],   #i: i+crop_height
                                       tileRange[1]:tileRange[3]]   #j: j + crop_width                                                
                crop_inital_diff = inital_diff[ tileRange[0]:tileRange[2],   #i: i+crop_height
                                                tileRange[1]:tileRange[3]]   #j: j + crop_width    
                crop_binary_target = binary_target[ tileRange[0]:tileRange[2],   #i: i+crop_height
                                       tileRange[1]:tileRange[3]]   #j: j + crop_width    
                crop_error_rate = crop_inital_diff.sum()/crop_binary_target.sum()* 100 
                if ( crop_target0.mean() > target0_mean/4 and  crop_error_rate> 1.2 and  inlier_rate < 0.4 and
                      crop_inital_diff.sum() >  ( paras.tile_shape[0] *paras.tile_shape[1]/64 ) ):
                    bsBoost_tileRange_ls.append(tileRange)                                    
            print ("bsBoost_tileRange_ls=",len(bsBoost_tileRange_ls))
                        
            diff_label_final = tiled_fcts. merge_diff_mask ( bsBoost_tileRange_ls, inital_diff,paras)
            
            if paras.demo == True:                         
                vis_diff =  vis_fcts.differenceVis (inital_diff ,dst, inliers, bsBoost_tileRange_ls, diff_label_final )                        
                print (" Before : error:",error)      
                vis_diff_resized = cv2.resize(img_as_ubyte(vis_diff), (vis_diff.shape[0]  // 2 , vis_diff.shape[1] // 2))
                io.imsave(os.path.join(output_dir, source_key.split(".")[0] + "-BeforeErr"+ '%.1f'%error+"%_diffVis.jpg"),
                            vis_diff_resized )        
#         else:
#             del inliers
            
        '''rerun the registration for bootstrap regions    '''    

        if bootstrap == True:            
            if diff_label_final.max() > 0 :
                for diff_label_obj in measure.regionprops(diff_label_final): 
                    mismatch_id = diff_label_obj.label
                    tileRange = diff_label_obj.bbox
                    ck_tileRange=  [ max( 0 , tileRange[0] - paras.ck_shift ), 
                                    max( 0 , tileRange[1] - paras.ck_shift ),
                                    min( int(paras.target_shape[0] ) -1, tileRange[2] + paras.ck_shift) , 
                                    min( int(paras.target_shape[1] ) -1, tileRange[3] + paras.ck_shift) ]         # extract the labels of subpicious window from previous merged results
                    ck_tileRange_shape = ( ck_tileRange[2] - ck_tileRange[0], ck_tileRange[3] - ck_tileRange[1] )
                    
                    target_tile =  target0[ ck_tileRange[0]:ck_tileRange[2],   #i: i+crop_height
                                            ck_tileRange[1]:ck_tileRange[3]]   #j: j + crop_width     # uint16                          
                    source_tile =  source[ ck_tileRange[0]:ck_tileRange[2],   #i: i+crop_height
                                            ck_tileRange[1]:ck_tileRange[3]]   #j: j + crop_width     # uint16                
                    fill_mask_tile = diff_label_final[ ck_tileRange[0]:ck_tileRange[2],   #i: i+crop_height
                                                    ck_tileRange[1]:ck_tileRange[3]]   #j: j + crop_width     # uint16
                    fill_mask_tile = (fill_mask_tile==mismatch_id)   # extrac the binmask tile
                                    
                    if s_i == 0:   # Only get models from target0 source 0 
                        if reuse_kp == False:
                            model_mini = mini_register(target_tile,source_tile,paras)    # extract the model inverse map from CHN0        
                        else:
                        
                            # load global keypoints
                            kp0_tile = select_keypointsInMask(keypoints0, (diff_label_final==mismatch_id))
                            kp1_tile = select_keypointsInMask(keypoints1, (diff_label_final==mismatch_id))                   
                            
                            # convert to local kps
                            kp0_crop = keypoints0[kp0_tile,:]
                            kp1_crop = keypoints1[kp1_tile,:]
                            kp0_crop = kp0_crop - np.tile([ck_tileRange[0], ck_tileRange[1]],
                                                            (kp0_crop.shape[0], 1))           # add the crop coordinate shift
                            kp1_crop = kp1_crop - np.tile([ck_tileRange[0], ck_tileRange[1]],
                                                            (kp1_crop.shape[0], 1))           # add the crop coordinate shift
                            model_mini = mini_register(target_tile,source_tile,paras , 
                                                    kp0_crop,descriptors0[kp0_tile,:],
                                                    kp1_crop,descriptors1[kp1_tile,:]  )    # extract the model inverse map from CHN0    
                        model_mini_dic[mismatch_id] = model_mini     # save the bootstrap transform into dict
                        
                    if model_mini_dic[mismatch_id]  is not None:  # only update when the model has been found                   
                        ## clear the old registered result       
                        source_warped[ ck_tileRange[0]:ck_tileRange[2],       
                                    ck_tileRange[1]:ck_tileRange[3]] = source_warped[ ck_tileRange[0]:ck_tileRange[2],       
                                                                                        ck_tileRange[1]:ck_tileRange[3]] *(1-fill_mask_tile)
                        # wrap the whole image    
                        source_warped_tile = warp( source_tile, 
                                                inverse_map = model_mini_dic[mismatch_id].inverse, 
                                                output_shape = ck_tileRange_shape)   
                        ## fill with the new one
                        source_warped[ ck_tileRange[0]:ck_tileRange[2],       
                                    ck_tileRange[1]:ck_tileRange[3]] += source_warped_tile*fill_mask_tile

        print ( "source_warped.type= ", source_warped.dtype, "range=",source_warped.max())
        t_ransac = time.time()

        ### save source
        output_type = input_type
        assert len( np.unique( source_warped ) ) > 1     # make sure the wrapped is not empty

        if input_type == np.uint8 or paras.demo:
            print("\tsaving image as 8 bit")
            tiff.imsave(os.path.join(output_dir, source_key),
                        np.clip(source_warped*255,0,255).astype( input_type) )  
        else:  # output_type is "16bit":
            print("\tsaving image as 16bit")
            tiff.imsave(os.path.join(output_dir, source_key),
                        np.clip(source_warped*65535,0,65535).astype( input_type))  

                 
        if s_i == 0 and paras.demo == True:            
            t_warp0 = time.time()
            print("[Timer] Image Warping for 1 channel ", source_key, " used time (h) =",
                  '%.2f'%((t_warp0 - t_ransac) / 3600))
            
            ########## save vis ############
            print("###########save vis")     
            vis, binary_diff,__,error,__= vis_fcts.eval_draw_diff (img_as_ubyte(target0),                                                            
                                                        img_as_ubyte(source_warped))              
            print (" After : error:",error)                    
            vis_resized = cv2.resize(img_as_ubyte(vis), (vis.shape[0] // 2, vis.shape[1] // 2))  
            io.imsave(os.path.join(output_dir, source_key.split(".")[0] + "_registeredVis.jpg"),
                         vis_resized)
            if bootstrap == False:
                vis_diff =  vis_fcts.differenceVis (binary_diff ,dst, inliers,
                                           tileRange_ls,diff_label_final)
            else:                
                vis_diff =  vis_fcts.differenceVis (binary_diff ,dst, inliers,
                                           bsBoost_tileRange_ls,diff_label_final)
                
            vis_diff_resized = cv2.resize(img_as_ubyte(vis_diff), (vis.shape[0]  // 2 , vis.shape[1] // 2))
            io.imsave(os.path.join(output_dir, source_key.split(".")[0] + "-Err"+ '%.1f'%error+"%_diffVis.jpg"),
                        vis_diff_resized )
        
    t_warp = time.time()
    print("[Timer] Image Warping for all channels used time (min) =",
          '%.2f'%((t_warp - t_ransac) / 60))        

    return keypoints1, descriptors1

def registration (input_dir,output_dir,target_round = "R2", imadjust = True,
                  multiprocess = True, keypoint_dir = None,bootstrap =False,
                  demo = False , tiling = "1000,1000" ,nKeypoint=  200000, 
                  ck_shift = 200, crop_overlap =50,residual_threshold = 5,min_samples = 6,
                  ransac_obj= "inlier_num",keypoint_tool = "cv2",match_perc=1.0,ds_rate =10):
    # Parameters


    print("Setting Parameteres:========")
    paras = Paras()
    paras.n_keypoints        = nKeypoint
    paras.ck_shift           = ck_shift
    paras.crop_overlap       = crop_overlap
    paras.min_samples        = min_samples
    paras.residual_threshold = residual_threshold
    paras.match_perc         = match_perc
    paras.ds_rate            = ds_rate

    paras.multiprocess  = False if str(multiprocess).lower() in ["f", "0", "false"] else multiprocess
    paras.keypoint_dir  = keypoint_dir
    paras.ransac_obj    = ransac_obj    
    paras.keypoint_tool = keypoint_tool
    paras.bootstrap     = str2bool(bootstrap)
    paras.demo          = str2bool(demo) 
    paras.imadjust      = str2bool(imadjust) 

    assert paras.keypoint_tool in ["cv2","skimage"]
    assert paras.min_samples >=3
    assert ( paras.match_perc>0 and paras.match_perc <=1.0)
    # import pdb;pdb.set_trace()
    # paras.display()
    if "," in tiling:
        paras.tile_shape = [int(s) for s in re.findall(r'\d+', tiling)]
    elif tiling == []:
        paras.tile_shape =[]  # no tiling, cautious might be extremely slow!

    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)           

    #    Set_name = os.listdir(args.input_dir)[1].split("_")[0] + "_"
    input_dir_image = [f for f in os.listdir(input_dir) if f.endswith('.tif')]
    Set_name = input_dir_image[0].split("_")[0] + "_"    
    Set_name = Set_name if "S" in Set_name else ""
    print ("Set_name=",Set_name)
    # the round that all other rounds are going to registered to !

    if paras.demo == False:
        print("Run all channels")
        # get all the channel and round id
        channels_range = []
        source_round_ls = []
        for fileName in sorted(input_dir_image):
            if "tif" in fileName and "C" in fileName:
                channel_id = fileName.split("C")[1].split(".")[0]
                if channel_id not in channels_range:
                    channels_range.append(channel_id)
                round_id = fileName.split("C")[0]
                if "_" in round_id:
                    round_id = round_id.split("_")[1]
                if round_id not in source_round_ls and round_id != target_round:
                    source_round_ls.append(round_id)
        print("channels_range=", channels_range)
        print("source_round_ls=", source_round_ls)
        
    else:  # only for testing or tunning the paras
        channels_range = [1]
        print("Run only C1 channel for all rounds")
        # paras.demo = True

        source_round_ls = []
        for fileName in sorted(input_dir_image)[:1]:
            if "tif" in fileName and "C" in fileName:
                channel_id = fileName.split("C")[1].split(".")[0]
                if channel_id not in channels_range:
                    channels_range.append(channel_id)
                round_id = fileName.split("C")[0]
                if "_" in round_id:
                    round_id = round_id.split("_")[1]
                if round_id not in source_round_ls and round_id != target_round:
                    source_round_ls.append(round_id)
        print("args.demo :", paras.demo, "\n Run for Channel ", channels_range, "\t Rounds: ", source_round_ls)

    ''' Run for images '''
    # target (reference image)
    targets = {}  # full filenames of all the images in target rounds

    for CHN in channels_range:
        target_fileName = Set_name + target_round + "C" + str(CHN) + ".tif"
        if os.path.isfile (os.path.join(input_dir, target_fileName )) == True:                                     # allow not continued channel iD               
            print("Read target image ", target_fileName)
            targets[target_fileName] = os.path.join(input_dir, target_fileName )
            shutil.copy (targets[   target_fileName] , os.path.join(output_dir, target_fileName))
        
    # source(image to be registered)
    for sr_i, source_round in enumerate(source_round_ls):
        sources = {}  # full filenames of all the images in target rounds
        print("*" * 10 + " " + source_round + "\n")
        for CHN in channels_range:        
            source_fileName = Set_name + source_round + "C" + str(CHN) + ".tif"
            source_fileDir  = os. path.join(input_dir , source_fileName)   
            if os.path.isfile (source_fileDir) == True:                                     # allow not continued channel iD               
                sources[source_fileName] =  source_fileDir
        # Run
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_target = True if sr_i == 0 else False  # only save target files in the last round
        

        if sr_i == 0:  # only need keypoint extraction for target for once
            keypoints1, descriptors1 = registrationORB_tiled(targets, sources, paras,                                                             
                                                            output_dir  = output_dir,
                                                            save_target = save_target,
                                                            )            
        else:
            _, _ = registrationORB_tiled(targets, sources, paras,
                                        output_dir = output_dir,
                                        save_target = save_target,

                                        keypoints1  = keypoints1,
                                        descriptors1= descriptors1,
                                       )

        print("\nRegistrationORB function for round", source_round, " finished!\n====================\n")

        
    print("\nRegistrationORB function finished!\n====================\n")
    print("Result in ", output_dir)

def str2bool(str_input):
    str_input = str(str_input) if type(str_input) != str else str_input
    bool_result = True if str_input.lower() in ["t", 'true', '1', "yes", 'y'] else False
    return bool_result

def main():
    parser = argparse.ArgumentParser(description='***  Whole brain segentation pipeline on DAPI + Histone Channel ***'
                                             + '\ne.g\n'
                                             + '\t$ python3 registration.py '
                                             + '-i $INPUT_DIR'
                                             + '-o $OUTPUT_DIR',
                                 formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-d', '--demo', default='False', type=str,required=False,
                        help=" 'T' only match channel 0, 'F' match all channels")
    parser.add_argument('-i', '--input_dir', 
                        default=r"D:\research in lab\dataset\50CHN\registration_demo\demo",
                        help='Path to the directory of input images')
    parser.add_argument('-o', '--output_dir',
                        default=r"D:\research in lab\dataset\50CHN\registration_demo\after",
                        help='Path to the directory to save output images')
    parser.add_argument('-nk', '--nKeypoint', required=False, default=200000, type=int,
                        help=" nKeypoint ")

    parser.add_argument('-kp_path', '--keypoint_dir', 
                        default= None, required=False,
                        help='Path to the directory to read the extracted keypoins')
    parser.add_argument('-mp', '--multiprocess', required=False, default="T", type=str,
                        help=" 'T' multi processing, 'F' single processing, 'N' specify number of threads", )
    parser.add_argument('--imadjust', required=False, default = 'T',type = str, 
                            help='whether to adjust the image for feature extraction')    
    parser.add_argument('--bootstrap', required=False, default = 'False',type = str, 
                            help='whether to adopt bootstrap to enhance registration')    
 
    parser.add_argument('--targetRound', required=False, default='R2', type=str,
                        help="keyword for target round")        
    parser.add_argument('-t', '--tiling', required=False, default="1000,1000", type=str,
                        help=" 'tiling_r, tilling_r' or '[]' or None, for 'user specified tiling shape', 'no tilling',or default ")
    parser.add_argument( '--ck_shift', required=False, default=200, type=int,
                        help=" check window dilation width for seaching robust keypoint")  # or 100
    parser.add_argument( '--crop_overlap', required=False, default=50, type=int,
                        help=" check window dilation width for seaching robust keypoint") # or 10
    parser.add_argument( '--residual_threshold', required=False, default=5, type=int,
                        help=" residual_threshold for ransac")
    parser.add_argument( '--ds_rate', required=False, default=10, type=int,
                        help=" downscale rate for fast performance  evaluation")
    parser.add_argument( '--min_samples', required=False, default=5, type=int,
                        help=" min_samples for ransac")
    parser.add_argument( '--match_perc', required=False, default=1.0, type=float,
                        help=" match_perc for matching keypoints")
    parser.add_argument( '--ransac_obj', required=False, type=str,default= "inlier_num",
                        help="evaluation function for ransac,in [ 'inlier_num' 'inlier_tile' 'residuals_sum' 'fast_diff']")                        
    parser.add_argument( '--keypoint_tool', required=False, type=str,default= "cv2",
                        help="package for keypoint extraction 'cv2' or 'skimage' ")      
    args = parser.parse_args()
    #%%
    tic = time.time()
    ############  

    registration (input_dir  = args.input_dir, output_dir = args.output_dir, # only those 2 are necessary
                  target_round          = args.targetRound,      
                  imadjust              = args.imadjust,   
                  multiprocess          = args.multiprocess, 
                  keypoint_dir          = args.keypoint_dir, 
                  bootstrap             = args.bootstrap,
                  demo                  = args.demo,  
                  tiling                = args.tiling ,
                  nKeypoint             = args.nKeypoint,
                  ck_shift              = args.ck_shift, 
                  crop_overlap          = args.crop_overlap,
                  residual_threshold    = args.residual_threshold,
                  min_samples           = args.min_samples,
                  ransac_obj            = args.ransac_obj,
                  keypoint_tool         = args.keypoint_tool,
                  match_perc            = args.match_perc,
                  ds_rate               = args.ds_rate
                  )

    toc = time.time()
    print("total time is (h) =", '%.2f'%((toc - tic) / 3600))


if __name__ == '__main__':

    start = time.time()
    main()
    print('*' * 50)
    print('*' * 50)
    print('Registeration pipeline finished successfully in {} seconds.'.format( time.time() - start))


