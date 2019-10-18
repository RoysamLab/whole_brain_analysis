'''
Image Registration of Channels for multi-rounds

Author: Rebecca LI, University of Houston, Farsight Lab, 2019
xiaoyang.rebecca.li@gmail.com

--- inspired  from ---
# https://github.com/scikit-image/skimage-tutorials/blob/master/lectures/adv3_panorama-stitching.ipynb
Improvements compared to above 
1) featureExtract_tiled (specifically for texture based large scale images)
2) support multiprocess & single thread
3) reuse of extracted feature in first rounds to the rest

--- conda packages ---
conda install -c conda-forge scikit-image \
conda install scikit-learn \
conda install -c conda-forge tifffile 

--- e.g. ---
&python registration_multiRds.py \
    -i [inputPath] \
    -o [outputPath]
'''

import os
import re
import sys
import time
import numpy as np
import shutil
import skimage,cv2
from skimage.color import rgb2gray
from skimage.feature import ORB, match_descriptors
from skimage import exposure,io
from skimage.transform import warp, ProjectiveTransform, SimilarityTransform, AffineTransform,PolynomialTransform,resize
from skimage.filters import threshold_otsu
from skimage.util import img_as_uint ,img_as_ubyte,img_as_float
import matplotlib.pyplot as plt
import tifffile as tiff
import multiprocessing
from multiprocessing.pool import ThreadPool
import warnings
warnings.filterwarnings("ignore")
import argparse
import random

from sklearn.neighbors import DistanceMetric

import tiled_mp_fcts as mpfcts
import vis_fcts as visfcts
from ransac_tile import ransac_tile

#import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='***  Whole brain segentation pipeline on DAPI + Histone Channel ***'
                                             + '\ne.g\n'
                                             + '\t$ python3 registrationORB_RANSIC.py  # run on whole brain \n '
                                             + '\t$ python3 registrationORB_RANSIC.py -t T  '
                                             + '-i /data/jjahanip/50_plex/stitched  '
                                             + '-w /data/xiaoyang/CHN50/Registered_Rebecca',
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-d', '--demo', default='F', type=str,
                    help=" 'T' only match channel 0, 'F' match all channels")
parser.add_argument('-i', '--input_dir', 
                    default=r"D:\research in lab\dataset\50CHN\registration_demo\before",
                    help='Path to the directory of input images')
parser.add_argument('-o', '--output_dir',
                    default=r"D:\research in lab\dataset\50CHN\registration_demo\after_overlap_new",
                    help='Path to the directory to save output images')
parser.add_argument('-ot', '--outputType', required=False, default="8bit", type=str,
                    help='Save tif image type: "8bit" or "16bit"')
parser.add_argument('-it', '--inputType', required=False, default="16bit", type=str,
                    help='Save tif image type: "8bit" or "16bit"')
parser.add_argument('-nk', '--nKeypoint', required=False, default=300000, type=int,
                    help=" nKeypoint ")
parser.add_argument('-kp_path', '--keypoint_dir', default=None,
                    help='Path to the directory to read the extracted keypoins')
parser.add_argument('-mt', '--maxtrials', required=False, default=5000, type=int,
                    help=" max_trials ")
parser.add_argument('-mp', '--multiprocess', required=False, default="4", type=str,
                    help=" 'T' multi processing, 'F' single processing")
parser.add_argument('--imadjust', required=False, default = 'F',type = str, 
                        help='whether to adjust the image for feature extraction')    
parser.add_argument('--targetRound', required=False, default='R2', type=str,
                    help="keyword for target round")        
# the smaller the tilling, longer the time, better the results
parser.add_argument('-t', '--tiling', required=False, default="2000,2000", type=str,
                    help=" 'tiling_r, tilling_r' or '[]' or None, for 'user specified tiling shape', 'no tilling',or default ")

args = parser.parse_args()

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        if type(array) == dict:
            text += ("dictionary len = " + len(array))
            for key in list(array.keys()):
                item = array[key]
                text += (" key = " + key + " \titem.shape= " + item.shape)
                text += (" min: {:10.5f}  max: {:10.5f}".format(item.min(), item.max()))
        else:
            text = text.ljust(25)
            text += ("shape: {:20}  ".format(str(array.shape)))
            if array.size:
                text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(), array.max()))
            else:
                text += ("min: {:10}  max: {:10}".exiformat("", ""))
            text += "  {}".format(array.dtype)
    print(text)


class Paras(object):
    def __init__(self):
        # Parameters for ORB
        self.n_keypoints = 300000
        self.fast_threshold = 0.08
        self.harris_k = 0.1

        # Parameters for skimage.measure.ransac
        self.min_samples = 5
        self.residual_threshold = 7  # default =1, 7
        self.max_trials = 600
        self.random_state = 42 #230

        # other user defined paras:
        self.target_shape = (4000,8000)
        self.tile_shape = (400, 800)        
        self.multiprocessing = False
        self.ck_shift = 50  # check window dilation width for seaching robust keypoint 
        self.crop_overlap = 10  
        self.err_thres = 1.5
        self.keypoint_dir = None
        

    def set_n_keypoints(self, n_keypoints):
        self.n_keypoints = n_keypoints

    def set_max_trials(self, max_trials):
        self.max_trials = max_trials

    def multiprocess(self, multiprocess):
        self.multiprocess = multiprocess

    def set_tile_shape(self, tile_shape):
        self.tile_shape = tile_shape

    def display(self):
        print("============================\n")
        print("Parameters for ORB : ")
        print("\tn_keypoints = ", self.n_keypoints)
        print("\tfast_threshold = ", self.fast_threshold)
        print("\tharris_k = ", self.harris_k)
        print("Parameters for skimage.measure.ransac : ")
        print("\tmin_samples = ", self.min_samples)
        print("\tresidual_threshold = ", self.residual_threshold)
        print("\tmax_trials = ", self.max_trials)
        print("\trandom_state = ", self.random_state)
        print("other user defined paras: ")
        print("\ttile_shape = ", self.tile_shape)
        print("\tmultiprocess = ", self.multiprocess)
        print("\ttarget_shape = ", self.target_shape)
        print("\tck_shift = ", self.ck_shift)
        print("\tcrop_overlap = ", self.crop_overlap)
        print("\tkeypoint_dir = ", self.keypoint_dir)

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



def mini_register(target,source,paras,src=None,dst = None):
    print ("\n#### mini_register",src,dst)   
    print ( "target.type= ", target.dtype, "target.mean=",target.mean() )
    print ( "source.type= ", target.dtype, "source.mean=",source.mean() )

#    import pdb ; pdb.set_trace()  
    if src is None:
        orb0 = ORB(n_keypoints = paras.n_keypoints,
                    fast_threshold = paras.fast_threshold,
                    harris_k = paras.harris_k)
        #  if without this step, orb.detect_and_extract will return error        
        orb0.detect(source)
    if dst is None:        
        orb1 = ORB(n_keypoints = paras.n_keypoints,
                    fast_threshold = paras.fast_threshold,
                    harris_k = paras.harris_k)   
        orb1.detect(target)                        # n by 256 array       
    
    print ("len(orb0.scales)= ",len(orb0.scales))
    print ("len(orb1.scales)= ",len(orb1.scales))

    if  len(orb0.scales) > 0 and len(orb1.scales) > 0:
        orb0.detect_and_extract(source)
        keypoints0 = orb0.keypoints  # n by 2 array, n is number of keypoints
        descriptors0 = orb0.descriptors                                        # n by 256 array
        orb1.detect_and_extract(target)
        keypoints1 = orb1.keypoints  # n by 2 array, n is number of keypoints
        descriptors1 = orb1.descriptors                

        matches01 = match_descriptors(descriptors0, descriptors1, metric ='hamming' , cross_check=True)
        # sort the matched points according to the distance
        src = keypoints0[matches01[:, 0]][:, ::-1]   # (image to be registered) 
        dst = keypoints1[matches01[:, 1]][:, ::-1]   #  (reference image, target) 
        
    model_robust01, inliers = skimage.measure.ransac((src, dst), ProjectiveTransform,
                                                min_samples         = min(paras.min_samples, matches01.shape[0]), 
                                                residual_threshold  = paras.residual_threshold,
                                                max_trials          = paras.max_trials,
                                                random_state        = paras.random_state,
                                                ) 

    print ("\t min-inliers%  =" , ( inliers.sum()/len(inliers)) *100 )
    return model_robust01
    
def registrationORB_tiled(targets, sources, paras, write_path, output_type="16bit", input_type="16bit",
                          save_target=True, 
                          keypoints1=None, descriptors1=None, imadjust = False, verbose =0):
    t_0 = time.time()

    target0 = []  # only use target 0 and source 0 for keypoint extraction
    source0 = []
    
    
    # READING IMAGES
    for key_id, t_key in enumerate( sorted ( targets.keys() )) :                
        # if "C0" in t_key:
        if key_id == 0:
            target0 = tiff.imread(targets[t_key])
            target0 = rgb2gray(target0) if target0.ndim == 3 else target0
            print("Process ", t_key, " as target0", "target0.shape = ", target0.shape)
            target0_key = t_key
    for key_id, s_key in enumerate( sorted ( sources.keys() )) :
        # if "C0" in s_key:
        if key_id == 0:
            source0 = tiff.imread(sources[s_key])
            source0 = rgb2gray(source0) if source0.ndim == 3 else source0
            print("Process ", s_key, " as source0", "source0.shape =", source0.shape)
            source0_key = s_key

    if target0 is [] or source0 is []:
        print("!! target or source not found error:", sys.exc_info()[0])

    '''1. Feature detection and matching'''
    target0 = img_as_ubyte(target0) if input_type is "8bit" else target0
    source0 = img_as_ubyte(source0) if input_type is "8bit" else source0

    target0 = visfcts.adjust_image (target0) if imadjust is True  else target0
    source0 = visfcts.adjust_image (source0) if imadjust is True else source0
    paras.target_shape = ( target0.shape[0],target0.shape[1] )
    
    paras.display()
    
    target0_mean = target0.mean()
    t_8bit = time.time()
    print("[Timer] Convert 16bit to 8bit used time (h) =", str((t_8bit - t_0) / 3600))
    
    # set tile ranges list 
    if paras.tile_shape == []:  # do not apply any tilling
        tile_shape = (int(target0.shape[0]), int(target0.shape[1]))
    else:  # default tile shape is 400*800
        tile_shape = paras.tile_shape
    tile_width, tile_height    = tile_shape
    img_rows = int(target0.shape[0])
    img_cols = int(target0.shape[1])                    

    tileRange_ls    = visfcts.crop_tiles(  ( img_rows,img_cols ),
                                           ( tile_width, tile_width),
                                           paras.crop_overlap)  
    paras.tiles_numbers = len(tileRange_ls)

    if verbose == 1 : 
        print ("tile_shape=" ,tile_shape)
        print("number of the tiles = ", paras.tiles_numbers)
        print("tileRange_ls[0] = ", tileRange_ls[0] )

    # EXTRACT KEYPOINTS
    if paras.keypoint_dir == None:
        keypoints0, descriptors0 = mpfcts.featureExtract_tiled(source0, paras, tileRange_ls)  # keypoints0.max(axis=1)
        if keypoints1 is None or descriptors1 is None:  # need to create featureExtraction for target, else read the created one from input
            keypoints1, descriptors1 = mpfcts.featureExtract_tiled(target0, paras, tileRange_ls)
        
        np.save( os.path.join ( write_path, source0_key.split(".")[0] + "_keypoints0.npy"))
        np.save( os.path.join ( write_path, source0_key.split(".")[0] + "_descriptors0.npy"))
        np.save( os.path.join ( write_path, target0_key.split(".")[0] + "_keypoints1.npy"))
        np.save( os.path.join ( write_path, target0_key.split(".")[0] + "_descriptors1.npy"))    
        print ("EXTRACT KEYPOINTS have been saved")

    else:
        keypoints0   = np.load( os.path.join ( write_path, source0_key.split(".")[0] + "_keypoints0.npy"))
        descriptors0 = np.load( os.path.join ( write_path, source0_key.split(".")[0] + "_descriptors0.npy"))
        keypoints1   = np.load( os.path.join ( write_path, target0_key.split(".")[0] + "_keypoints1.npy"))
        descriptors1 = np.load( os.path.join ( write_path, target0_key.split(".")[0] + "_descriptors1.npy"))           
        print ("EXTRACT KEYPOINTS have been load from path")

    t_featureExtract_tiled = time.time()
    print("[Timer] featureExtract tiled used time (h) =", str((t_featureExtract_tiled - t_8bit) / 3600))
    
    # Match descriptors between target and source image
    if paras.multiprocessing == False:
        src,dst =  mpfcts.match_descriptors_tiled(keypoints0,descriptors0,
                                          keypoints1,descriptors1, 
                                          paras, tileRange_ls ,
                                          ck_shift = paras.ck_shift)
    else:
        src,dst =  mpfcts.transfromest_tiled (keypoints0,descriptors0,
                                          keypoints1,descriptors1, 
                                          paras, tileRange_ls ,
                                          ck_shift = paras.ck_shift)

    ''' 2. Transform estimation ''' 
    print ("match_descriptors:", src.shape)  # N * 2
    
    model_robust01, inliers = skimage.measure.ransac((src, dst), ProjectiveTransform,
                                                min_samples         = paras.min_samples, 
                                                residual_threshold  = paras.residual_threshold,
                                                max_trials          = paras.max_trials,
                                                random_state        = paras.random_state,
                                                )
    print ("\t Inital inliers%  =" , ( inliers.sum()/len(inliers)) *100 )

    exam_tileRange_ls = []
    for i in range(0, img_rows, int( tile_height/3)):
       for j in range(0, img_cols, int( tile_width/3)):
           tileRange = [ i,j,
                          min(img_rows, tile_height + i),
                          min(img_cols, tile_width  + j )]
           exam_tileRange_ls.append ( tileRange )
    spl_tile_ls, spl_tile_dic = spl_tiled_data ((src, dst) , exam_tileRange_ls)

    model_robust01, inliers = ransac_tile((src, dst), AffineTransform,
                                                min_samples         = paras.min_samples, 
                                                residual_threshold  = paras.residual_threshold,
                                                max_trials          = paras.max_trials,
                                                random_state        = paras.random_state,
                                                spl_tile_ls = spl_tile_ls)      
    print ("\t Final inliers%  =" , ( inliers.sum()/len(inliers)) *100 )
#    
    boostrapped = True
    if boostrapped == True:
#        boostrapped_dic = {}
        boostrap_tileRange_ls = []
        for tileRange_key in spl_tile_dic.keys() :
            tileRange = exam_tileRange_ls[tileRange_key]            #exam tile range (small)
            spl_tile =  spl_tile_dic[tileRange_key]
            inlier_tile = inliers [ spl_tile]
            inlier_rate = inlier_tile.sum() / len(inlier_tile)          
            
            crop_target0= target0[ tileRange[0]:tileRange[2],   #i: i+crop_height
                                tileRange[1]:tileRange[3]]   #j: j + crop_width              
                      
            if (  len(inlier_tile) > paras.min_samples and inlier_rate < 0.07 and crop_target0.mean() >target0_mean/8  ):
                boostrap_tileRange_ls.append(tileRange)
                
#        print ("boostrap_tileRange_ls =" ,boostrap_tileRange_ls)        
        boostrap_tileRange_merged_ls = []
        # Merge boostrap_tileRange_ls  , output : boostrap_tileRange_merged_ls
        len_diff = len (boostrap_tileRange_ls) - 0
        boostrap_tileRange_to_merge_ls = boostrap_tileRange_ls.copy()
        while len_diff !=0  :      
            boostrap_tileRange_merged_ls = visfcts.merge_tileRange(boostrap_tileRange_to_merge_ls)   
            print ("before len =", len ( boostrap_tileRange_to_merge_ls), 
                   "after len =",  len ( boostrap_tileRange_merged_ls) )
            len_diff = len ( boostrap_tileRange_to_merge_ls) -len (boostrap_tileRange_merged_ls)
            boostrap_tileRange_to_merge_ls = boostrap_tileRange_merged_ls

        boostrap_spl_tile_ls = spl_tiled_data ((src, dst) , boostrap_tileRange_merged_ls)   # get keypoint idxs
        
    del descriptors0,keypoints0
    
    ''' 3. Image Warping'''
    # we must warp, or transform, two of the three images so they will properly align with the stationary image.
    
    # Apply same offset on all rest images       
    
    for s_i, source_key in enumerate( sorted ( sources.keys() )) :
        # if "C0" in s_key:
        source = tiff.imread(sources[source_key])
        source = rgb2gray(source) if source.ndim == 3 else source               # uint16
        
        source = visfcts.check_shape(source,paras.target_shape)                        
        source_warped = warp(source, inverse_map = model_robust01.inverse, 
                                 output_shape = paras.target_shape)             # float64
        
        # rerun the registration for boostrapped regions
        if boostrapped == True:

            # clear the old registered result       
            if len(boostrap_tileRange_merged_ls) > 0 :
                for i, tileRange in enumerate( boostrap_tileRange_merged_ls): 
                    boostrap_tileRange_merged_ls[i] = tileRange
                    source_warped[ tileRange[0]:tileRange[2],   #i: i+crop_height
                                   tileRange[1]:tileRange[3]] =  0   
            # fill with the new one
            if len(boostrap_tileRange_merged_ls) > 0 :
                for i, tileRange in enumerate( boostrap_tileRange_merged_ls): 
                    print ("Wrap: tileRange=",tileRange)   
                    ck_tileRange=  [ max( 0 , tileRange[0] - paras.ck_shift ), 
                                     max( 0 , tileRange[1] - paras.ck_shift ),
                                     min( int(paras.target_shape[0] ) -1, tileRange[2] + paras.ck_shift) , 
                                     min( int(paras.target_shape[1] ) -1, tileRange[3] + paras.ck_shift) ]         # extract the labels of subpicious window from previous merged results
                    
                    target_tile =  target0[ tileRange[0]:tileRange[2],   #i: i+crop_height
                                            tileRange[1]:tileRange[3]]   #j: j + crop_width     # uint16                             
                                           
                    source_tile =  source[ ck_tileRange[0]:ck_tileRange[2],   #i: i+crop_height
                                           ck_tileRange[1]:ck_tileRange[3]]   #j: j + crop_width     # uint16

                    if s_i == 0:        
                        source0_tile = source0[ ck_tileRange[0]:ck_tileRange[2],   #i: i+crop_height
                                           ck_tileRange[1]:ck_tileRange[3]]   #j: j + crop_width     # uint16   
                        model_mini = mini_register(target_tile,source0_tile,paras)    # extract the model inverse map from CHN0             
                    
                    source_warped_tile = warp(source_tile, inverse_map = model_mini.inverse, 
                                              output_shape = target_tile.shape)         
                    source_warped_before =  source_warped[ tileRange[0]:tileRange[2],   #i: i+crop_height
                                                           tileRange[1]:tileRange[3]]
                    fill_mask = (source_warped_before > 0)
                    source_warped_tile[fill_mask] = 0                           # remove the registered pixels in previous step
                    source_warped[ tileRange[0]:tileRange[2],       
                                   tileRange[1]:tileRange[3]] += source_warped_tile


#        import pdb ; pdb.set_trace()         
        print ( "source_warped.type= ", source_warped.dtype, "source_warped.max=",source_warped.max )
        source_warped = img_as_uint(source_warped).astype( target0.dtype)                  
        t_ransac = time.time()

        ### save source
        if output_type is "8bit":
            tiff.imsave(os.path.join(write_path, source_key),
                        img_as_ubyte(source_warped), bigtiff=True)
        else:  # output_type is "16bit":
            print("\t\t saving image as 16bit")
            tiff.imsave(os.path.join(write_path, source_key),
                         img_as_uint(source_warped), bigtiff=True)                   
        if s_i == 0:            
            t_warp0 = time.time()
            print("[Timer] Image Warping for 1 channel ", source_key, " used time (h) =",
                  str((t_warp0 - t_ransac) / 3600))
            
            ########## save vis ############
            print("###########save vis")     
     
            vis, binary_diff,error,__ = visfcts.eval_draw_diff (img_as_ubyte(target0),                                                            
                                                        img_as_ubyte(source_warped)) 
            print (" After : error:",error)                    
            vis_resized = resize(vis, (vis.shape[0] // 2, vis.shape[1] // 2))    
            io.imsave(os.path.join(write_path, source_key.split(".")[0] + "_registeredVis.jpg"),
                         vis_resized)
            if boostrapped == False:
                vis_diff =  visfcts.differenceVis (binary_diff ,dst, inliers,
                                           tileRange_ls)
            else:                
                vis_diff =  visfcts.differenceVis (binary_diff ,dst, inliers,
                                           boostrap_tileRange_ls,boostrap_tileRange_merged_ls)
                
            vis_diff_resized = resize(vis_diff, (vis.shape[0]  // 2 , vis.shape[1] // 2) ,
                                                    mode = "constant")
            print ("########### error = ", error)
            io.imsave(os.path.join(write_path, source_key.split(".")[0] + "-Err"+ '%.1f'%error+"%_diffVis.jpg"),
                        vis_diff_resized )
        
    t_warp = time.time()
#    print("sources_warped.shape= ", source_warped.shape)  
    print("[Timer] Image Warping for all channels used time (h) =",
          str((t_warp - t_ransac) / 3600))
        


    return keypoints1, descriptors1


def str2bool(str_input):
    bool_result = True if str_input.lower() in ["t", 'true', '1', "yes", 'y'] else False
    return bool_result

def main():

    tic = time.time()

    ############  

    # Parameters
    print("Reading Parameteres:========")
    paras = Paras()
    paras.set_n_keypoints(args.nKeypoint)
    paras.set_max_trials(args.maxtrials)

    multiprocess = False if args.multiprocess.lower() in ["f", "0", "false"] else args.multiprocess
    paras.multiprocess(multiprocess)
    paras.keypoint_dir = args.keypoint_dir
    
    if "," in args.tiling:
        set_tile_shape = [int(s) for s in re.findall(r'\d+', args.tiling)]
        paras.set_tile_shape(set_tile_shape)
    elif args.tiling == []:
        paras.set_tile_shape([])  # no tiling, cautious might be extremely slow!
    # else is [defalut 400,600]    
    write_path = args.output_dir
    if os.path.exists(write_path) is False:
        os.mkdir(write_path)        

    assert args.outputType in ["16bit", "8bit"]

    Set_name = os.listdir(args.input_dir)[1].split("_")[0] + "_"
    Set_name = Set_name if "S" in Set_name else ""
    # the round that all other rounds are going to registered to !
    target_round = args.targetRound #"R2"

    if str2bool(args.demo) is False:
        print("Run all channels")
        # get all the channel and round id
        channels_range = []
        source_round_ls = []
        for fileName in sorted(os.listdir(args.input_dir)):
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
        source_round_ls = []
        for fileName in sorted(os.listdir(args.input_dir))[:1]:
            if "tif" in fileName and "C" in fileName:
                channel_id = fileName.split("C")[1].split(".")[0]
                if channel_id not in channels_range:
                    channels_range.append(channel_id)
                round_id = fileName.split("C")[0]
                if "_" in round_id:
                    round_id = round_id.split("_")[1]
                if round_id not in source_round_ls and round_id != target_round:
                    source_round_ls.append(round_id)
        print("args.demo :", args.demo, "\n Run for Channel ", channels_range, "\t Rounds: ", source_round_ls)

    ''' Run for images '''
    # target (reference image)
    targets = {}  # full filenames of all the images in target rounds

    for CHN in channels_range:
        target_fileName = Set_name + target_round + "C" + str(CHN) + ".tif"
        print("Read target image ", target_fileName)
        targets[target_fileName] = os.path.join(args.input_dir, target_fileName)
        shutil.copy (targets[target_fileName] , os.path.join(args.output_dir, target_fileName))
        
    # source(image to be registered)
    for sr_i, source_round in enumerate(source_round_ls):
        sources = {}  # full filenames of all the images in target rounds
        print("*" * 10 + " " + source_round + "\n")

        for CHN in channels_range:        
            source_fileName = Set_name + source_round + "C" + str(CHN) + ".tif"
            source_fileDir  = os. path.join(args.input_dir , source_fileName)   
            print (" source_fileDir = " , source_fileDir)
            if os.path.isfile (source_fileDir) == True:                                     # allow not continues channel iD               
                print("Read source image  ",source_fileName)
                sources[source_fileName] =  source_fileDir

        # Run
        if not os.path.exists(write_path):
            os.makedirs(write_path)

        save_target = True if sr_i == 0 else False  # only save target files in the last round

        if sr_i == 0:  # only need keypoint extraction for target for once
            keypoints1, descriptors1 = registrationORB_tiled(targets, sources, paras,
                                                            write_path=write_path,
                                                            output_type=args.outputType,
                                                            input_type=args.inputType,
                                                            save_target=save_target,
                                                            imadjust= str2bool(args.imadjust) )            
        else:
            _, _ = registrationORB_tiled(targets, sources, paras,
                                        write_path=write_path,
                                        output_type=args.outputType,
                                        input_type=args.inputType,
                                        save_target=save_target,
                                        imadjust=str2bool(args.imadjust),
                                        keypoints1=keypoints1,
                                        descriptors1=descriptors1)

        print("\nRegistrationORB function for round", source_round, " finished!\n====================\n")

        
    print("\nRegistrationORB function finished!\n====================\n")
    print("Result in ", write_path)

    toc = time.time()
    print("total time is (h) =", str((toc - tic) / 3600))


if __name__ == '__main__':

    start = time.time()
    main()
    print('*' * 50)
    print('*' * 50)
    print('Registeration pipeline finished successfully in {} seconds.'.format( time.time() - start))



#
#
#
#binary_diff = io.imread(r"D:\research in lab\dataset\50CHN\registration_demo\after\R1C1-Err3.9%_differenceVis.jpg")[:100,:100,0]
#target_img = io.imread(r"D:\research in lab\dataset\50CHN\registration_demo\after\R2C1.tif")
#binary_target = target_img >= threshold_otsu(target_img) * 1.2
#binary_diff = (binary_diff > 125)*1
#
##plt.figure(),plt.imshow(binary_target)
##plt.figure(),plt.imshow( ,plt.colorbar()
#
#tile_width, tile_height    = [1000,1000]
#img_rows,img_cols = (10000, 8000)                   
#tileRange_ls = []    
#crop_overlap = 10
#for i in range(0, img_rows, tile_height- crop_overlap ):
#   for j in range(0, img_cols, tile_width -crop_overlap):
#       tileRange_ls.append ( [ i,j,
#                              min(img_rows, tile_height + i),
#                              min(img_cols, tile_width  + j )] )
#tiles_numbers = len(tileRange_ls)
#    
#diff_area = np.zeros( [int(img_rows/tile_height) , int(img_cols/tile_width)] )
#for t_i, tileRange in enumerate( tileRange_ls):    
#    crop_diff = binary_diff [ tileRange[0]:tileRange[2], 
#                              tileRange[1]:tileRange[3]] 
#    crop_target = binary_target [ tileRange[0]:tileRange[2], 
#                                  tileRange[1]:tileRange[3]] 
#    diff = crop_diff.sum()/crop_target.sum()
#    i , j = [ int(tileRange[0]/tile_height), int(tileRange[1]/tile_width)]
#    diff_area[i,j] = diff
#plt.imshow( diff_area),plt.colorbar()
#

