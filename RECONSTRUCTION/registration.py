'''
Image Registration of Channels for multi-rounds

Author: Rebecca LI, University of Houston, Farsight Lab, 2019
xiaoyang.rebecca.li@gmail.com

--- inspired  from ---
# https://github.com/scikit-image/skimage-tutorials/blob/master/lectures/adv3_panorama-stitching.ipynb
Improvements compared to above 
1) featureExtract_tiled (specifically for texture based large scale images)
2) support multiprocessing & single thread
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
import skimage
from skimage.color import rgb2gray
from skimage.feature import ORB, match_descriptors
from skimage import exposure
from skimage.transform import warp, ProjectiveTransform, SimilarityTransform
import tifffile as tiff
import multiprocessing
from multiprocessing.pool import ThreadPool
import warnings
warnings.filterwarnings("ignore")
import argparse

parser = argparse.ArgumentParser(description='***  Whole brain segentation pipeline on DAPI + Histone Channel ***'
                                             + '\ne.g\n'
                                             + '\t$ python3 registrationORB_RANSIC.py  # run on whole brain \n '
                                             + '\t$ python3 registrationORB_RANSIC.py -t T  '
                                             + '-i /data/jjahanip/50_plex/stitched  '
                                             + '-w /data/xiaoyang/CHN50/Registered_Rebecca',
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-d', '--demo', default='T', type=str,
                    help=" 'T' only match channel 0, 'F' match all channels")
parser.add_argument('-i', '--input_dir', 
                    default=r"/data/xiaoyang/80-plex/1_GFP-NSC_WT-pMCAO_3d1w/tif",
                    help='Path to the directory of input images')
parser.add_argument('-o', '--output_dir',
                    default=r"/data/xiaoyang/80-plex/1_GFP-NSC_WT-pMCAO_3d1w/registered_demo",
                    help='Path to the directory to save output images')
parser.add_argument('-ot', '--outputType', required=False, default="16bit", type=str,
                    help='Save tif image type: "8bit" or "16bit"')
parser.add_argument('-it', '--inputType', required=False, default="16bit", type=str,
                    help='Save tif image type: "8bit" or "16bit"')
parser.add_argument('-nk', '--nKeypoint', required=False, default=300000, type=int,
                    help=" nKeypoint ")
parser.add_argument('-mt', '--maxtrials', required=False, default=600, type=int,
                    help=" max_trials ")
parser.add_argument('-mp', '--multiprocessing', required=False, default='T', type=str,
                    help=" 'T' multi processing, 'F' single processing")
parser.add_argument('--imadjust', required=False,
                        default = '0',type = str, 
                        help='whether to adjust the image for feature extraction')            
# the smaller the tilling, longer the time, better the results
parser.add_argument('-t', '--tiling', required=False, default="400,800", type=str,
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
        self.min_samples = 4
        self.residual_threshold = 1
        self.max_trials = 600
        self.random_state = 230

        # other user defined paras:
        self.tile_shape = (400, 800)
        self.multiprocessing = False

    def set_n_keypoints(self, n_keypoints):
        self.n_keypoints = n_keypoints

    def set_max_trials(self, max_trials):
        self.max_trials = max_trials

    def set_multiprocessing(self, multiprocessing):
        self.multiprocessing = multiprocessing

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
        print("\tmultiprocessing = ", self.multiprocessing)

def featureExtract_tiled(img, paras, tileRange_ls ):
    if paras.multiprocessing == False:  # use single process option
        keypoint_all, descriptor_all = orb_detect_tiled(img, paras, tileRange_ls)

    else:                              # use multiprocess option
        # Set numOfThreads
        if paras.multiprocessing.isnumeric():
            print("&&&&&& Use specified number of numOfThreads ")
            numOfThreads = paras.multiprocessing
        else:
            numOfThreads = multiprocessing.cpu_count()
            print("&&&&&&& Use defalut number of numOfThreads")
        print("numOfThreads = ", numOfThreads)        
        pool = ThreadPool(processes=numOfThreads)
        ls_size = int(np.ceil(len(tileRange_ls)/numOfThreads))
        print ("ls_size =",ls_size)

        # run multiprocessin
        keypoint_alltiles   = np.zeros((1,2))
        descriptor_alltiles = np.zeros((1,256))
        async_result = []

        for th in range (0, numOfThreads):
            tileRange_ls_mp = tileRange_ls[  th*ls_size: th*ls_size +ls_size ]                                      # split the whole tileRange_ls in to parts for multiprocessing  
            async_result.append(  pool.apply_async(orb_detect_tiled, ( img, paras, tileRange_ls_mp )))  # tuple of args for foo
            print("\tmulti thread for", th, " ... ,","len(tileRange_ls_mp)=",len(tileRange_ls_mp),"\n")

        # load results
        for r in async_result:
            keypoint_alltiles   = np.concatenate((keypoint_alltiles,   r.get()[0]), axis=0)
            descriptor_alltiles = np.concatenate((descriptor_alltiles, r.get()[1]), axis=0)

        print("[multiprocessing] featureExtract_tiled: keypoint_alltiles.shape = ", keypoint_alltiles.shape)
        keypoint_all   =  keypoint_alltiles[1:, :]
        descriptor_all =  descriptor_alltiles[1:, :]
    
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

    for tileRange in tileRange_ls:
        crop_img = img[ tileRange[0]:tileRange[2],   #i: i+crop_height
                        tileRange[1]:tileRange[3]]   #j: j + crop_width     
        if verbose == 1:
            print ("$"*5 + "[Debug point 1]","tileRange=", tileRange )
            log(crop_img)
        # only tiles contain enough engergy (obvious contrast of adjacent pixels) have keypoints                
        if skimage.measure.shannon_entropy(crop_img) > 5:  # >10
            tile_n_keypoints = int(paras.n_keypoints / paras.tiles_numbers)
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
                                                (keypoints0.shape[0], 1))  # add the crop coordinate shift
                descriptor = orb.descriptors  # n by 256 array

                keypoint_alltiles   = np.concatenate((keypoint_alltiles , keypoint), axis=0)
                descriptor_alltiles = np.concatenate((descriptor_alltiles, descriptor), axis=0)

    keypoint_all    = keypoint_alltiles     [1:, :]
    descriptor_all  = descriptor_alltiles   [1:, :]

    if verbose == 1:
        print("*" * 10 + "orb_detect_tiled : = ", keypoint_alltiles.shape[0])
    return keypoint_all, descriptor_all


def offset_calculation(pano1, pano0, model_robust01):
    # Shape of t image, our registration target
    r, c = pano1.shape[:2]
    # Note that transformations take coordinates in (x, y) format,
    # not (row, column), in order to be consistent with most literature
    corners = np.array([[0, 0], [0, r], [c, 0], [c, r]])

    # Warp the image corners to their new positions
    warped_corners01 = model_robust01(corners)
    all_corners = np.vstack((warped_corners01, corners))

    # The overall output shape will be max - min
    corner_min = np.min(all_corners, axis=0)
    corner_max = np.max(all_corners, axis=0)
    output_shape = (corner_max - corner_min)

    # Ensure integer shape with np.ceil and dtype conversion
    output_shape = np.ceil(output_shape[::-1]).astype(int)

    # Apply estimated transforms

    # This in-plane offset is the only necessary transformation for the middle image
    offset1 = SimilarityTransform(translation=-corner_min)

    return offset1, output_shape


def image_warping(pano1, pano0, offset1, model_robust01, output_shape):
    # Translate pano1 into place
    pano1_warped = warp(pano1, offset1.inverse, order=3,
                        output_shape=output_shape, cval=-1)

    # Acquire the image mask for later use
    pano1_mask = (pano1_warped != -1)  # Mask == 1 inside image
    pano1_warped[~pano1_mask] = 0  # Return background values to 0

    # Warp pano0 to pano1 (target)
    transform01 = (model_robust01 + offset1).inverse
    pano0_warped = warp(pano0, transform01, order=3,
                        output_shape=output_shape, cval=-1)

    pano0_mask = (pano0_warped != -1)  # Mask == 1 inside image
    pano0_warped[~pano0_mask] = 0  # Return background values to 0

    croppedCoords_r, croppedCoords_c = np.where(pano1_mask == 1)
    pano1_warped_cropped = pano1_warped[croppedCoords_r.min():croppedCoords_r.max(),
                           croppedCoords_c.min():croppedCoords_c.max()]
    pano0_warped_cropped = pano0_warped[croppedCoords_r.min():croppedCoords_r.max(),
                           croppedCoords_c.min():croppedCoords_c.max()]

    return pano1_warped_cropped, pano0_warped_cropped

def adjust_image (img):
    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))          # Load over images
    return img

def registrationORB_tiled(targets, sources, paras, write_path, output_type="16bit", input_type="16bit",
                          save_target=True, keypoints1=None, descriptors1=None, imadjust = False, verbose =0):
    t_0 = time.time()

    target0 = []  # only use target 0 and source 0 for keypoin extraction
    source0 = []
    # READING IMAGES
    for t_key in targets.keys():
        # if "C0" in t_key:
        if "C1.tif" in t_key:
            target0 = tiff.imread(targets[t_key])
            target0 = rgb2gray(target0) if target0.ndim == 3 else target0
            print("Process ", t_key, " as target0", "target0.shape = ", target0.shape)
    for s_key in sources.keys():
        # if "C0" in s_key:
        if "C1.tif" in s_key:
            source0 = tiff.imread(sources[s_key])
            source0 = rgb2gray(source0) if source0.ndim == 3 else source0
            print("Process ", s_key, " as source0", "source0.shape =", source0.shape)

    if target0 is [] or source0 is []:
        print("!! target or source not found error:", sys.exc_info()[0])

    '''1. Feature detection and matching'''
    target0 = skimage.util.img_as_ubyte(target0) if input_type is "8bit" else target0
    source0 = skimage.util.img_as_ubyte(source0) if input_type is "8bit" else source0
    print("source0.max()=", source0.max())

    target0 = adjust_image (target0) if imadjust is True else target0
    source0 = adjust_image (source0) if imadjust is True else source0

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
    tileRange_ls = []    
    for i in range(0, img_rows, tile_height ):
       for j in range(0, img_cols, tile_width ):
           tileRange_ls.append ( [ i,j,tile_height + i, tile_width + j] )
    paras.tiles_numbers = len(tileRange_ls)
    
    if verbose == 1 : 
        print ("tile_shape=" ,tile_shape)
        print("number of the tiles = ", paras.tiles_numbers)
        print("tileRange_ls[0] = ", tileRange_ls[0] )


    # EXTRACT KEYPOINTS
    keypoints0, descriptors0 = featureExtract_tiled(source0, paras, tileRange_ls)
    if keypoints1 is None or descriptors1 is None:  # need to create featureExtraction for target, else read the created one from input
        keypoints1, descriptors1 = featureExtract_tiled(target0, paras, tileRange_ls)

    t_featureExtract_tiled = time.time()
    print("[Timer] featureExtract tiled used time (h) =", str((t_featureExtract_tiled - t_8bit) / 3600))


    # Match descriptors between left/right images and the center
    matches01 = match_descriptors(descriptors0, descriptors1, cross_check=True)

    print("len(matches01)= ", len(matches01))

    ''' 2. Transform estimation '''
    # Select keypoints from
    #   * source (image to be registered)   : pano0
    #   * target (reference image)          : pano1, our middle frame registration target
    src = keypoints0[matches01[:, 0]][:, ::-1]
    dst = keypoints1[matches01[:, 1]][:, ::-1]

    model_robust01, __ = skimage.measure.ransac((src, dst), ProjectiveTransform,
                                                min_samples=paras.min_samples,
                                                residual_threshold=paras.residual_threshold,
                                                max_trials=paras.max_trials,
                                                random_state=paras.random_state)
    t_ransac = time.time()
    print("[Timer] Transform estimation/skimage.measure.ransac used time (h) =",
          str((t_ransac - t_featureExtract_tiled) / 3600))

    ''' 3. Image Warping'''
    # we produce the panorama itself.
    # we must warp, or transform, two of the three images so they will properly align with the stationary image.
    offset1, output_shape = offset_calculation(target0, source0, model_robust01)

    # Apply same offset on all rest images       
    for s_i, (target_key, source_key) in enumerate(zip(targets.keys(), sources.keys())):
        pano1 = tiff.imread(targets[target_key])
        pano0 = tiff.imread(sources[source_key])
        pano1 = rgb2gray(pano1) if pano1.ndim == 3 else pano1
        pano0 = rgb2gray(pano0) if pano0.ndim == 3 else pano0

        pano1_warped_cropped, pano0_warped_cropped = image_warping(pano1, pano0,
                                                                   offset1, model_robust01,
                                                                   output_shape)
        print("pano1_warped_cropped.shape = ", pano1_warped_cropped.shape)
        print("pano0_warped_cropped.shape = ", pano0_warped_cropped.shape)

        wrap_size = pano1_warped_cropped.shape

        ### save target
        print("save_target = ", save_target)
        if save_target is True:
            if output_type is "8bit":
                tiff.imsave(os.path.join(write_path, target_key),
                            skimage.util.img_as_ubyte(pano1_warped_cropped), bigtiff=True)
            else:  # output_type is "16bit":
                print("\t\t saving image as 16bit")
                tiff.imsave(os.path.join(write_path, target_key),
                            skimage.util.img_as_uint(pano1_warped_cropped), bigtiff=True)
        ### save source
        if output_type is "8bit":
            tiff.imsave(os.path.join(write_path, source_key),
                        skimage.util.img_as_ubyte(pano0_warped_cropped), bigtiff=True)
        else:  # output_type is "16bit":
            print("\t\t saving image as 16bit")
            tiff.imsave(os.path.join(write_path, source_key),
                         skimage.util.img_as_uint(pano0_warped_cropped), bigtiff=True)
        if s_i == 0:
            t_warp0 = time.time()
            print("[Timer] Image Warping for 1 channel ", source_key, " used time (h) =",
                  str((t_warp0 - t_ransac) / 3600))

    t_warp = time.time()
    print("[Timer] Image Warping for all channels used time (h) =",
          str((t_warp - t_ransac) / 3600))

    return wrap_size, keypoints1, descriptors1


def wrap_size_check(input_LOC, write_LOC, wrap_size_min=[]):
    '''find all Base channel and crop them to the smallest wrap size'''
    # if when wrap_size_min is not set, find the smallest image size as wrap size    
    if wrap_size_min == []:
        wrap_size = [10000000, 10000000]  # the final size of image, take the smallest ones
        for fileName in os.listdir(input_LOC):
            if "C1.tif" in fileName:
                base_image = tiff.imread(os.path.join(input_LOC, fileName))
                wrap_size[0] = base_image.shape[0] if base_image.shape[0] < wrap_size[0] else wrap_size[0]
                wrap_size[1] = base_image.shape[1] if base_image.shape[1] < wrap_size[1] else wrap_size[1]
        print("wrap_size=", wrap_size)
        wrap_size_min = wrap_size

    ### crop all into smallest wrap size
    wrap_size = wrap_size_min
    for fileName in os.listdir(input_LOC):
        if ".tif" in fileName:
            image_to_save = tiff.imread(os.path.join(input_LOC, fileName))
            image_to_save = rgb2gray(image_to_save) if image_to_save.ndim == 3 else image_to_save

            image_to_save = image_to_save[0:wrap_size[0], 0:wrap_size[1]]
            tiff.imsave(os.path.join(write_LOC, fileName), image_to_save, bigtiff=True)  # make sure they are all same size


def str2bool(str_input):
    bool_result = True if str_input.lower() in ["t", 'true', '1', "yes", 'y'] else False
    return bool_result

def main():

    tic = time.time()

    # Parameters
    print("Reading Parameteres:========")
    paras = Paras()
    paras.set_n_keypoints(args.nKeypoint)
    paras.set_max_trials(args.maxtrials)

    multiprocessing = False if args.multiprocessing.lower() in ["f", "0", "false"] else args.multiprocessing
    paras.set_multiprocessing(multiprocessing)

    if "," in args.tiling:
        set_tile_shape = [int(s) for s in re.findall(r'\d+', args.tiling)]
        paras.set_tile_shape(set_tile_shape)
    elif args.tiling == []:
        paras.set_tile_shape([])  # no tiling, cautious might be extremely slow!
    # else is [defalut 400,600]    
    paras.display()
    write_path = args.output_dir

    assert args.outputType in ["16bit", "8bit"]

    Set_name = os.listdir(args.input_dir)[2].split("_")[0] + "_"
    Set_name = Set_name if "S" in Set_name else ""
    # the round that all other rounds are going to registered to !
    target_round = "R2"

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
        print("args.demo :", args.demo, "\n Run for Channel ", channels_range, "\t Rounds: ", source_round_ls)

    ''' Run for images '''
    # target (reference image)
    targets = {}  # full filenames of all the images in target rounds

    for CHN in channels_range:
        target_fileName = Set_name + target_round + "C" + str(CHN) + ".tif"
        print("Read target image ", target_fileName)
        targets[target_fileName] = os.path.join(args.input_dir, target_fileName)

    # source(image to be registered)
    wrap_size_min = [10000000, 10000000]  # the final size of image, take the smallest ones

    for s_i, source_round in enumerate(source_round_ls):
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

        save_target = True if s_i == 0 else False  # only save target files in the last round

        if s_i == 0:  # only need keypoint extraction for target for once
            wrap_size, keypoints1, descriptors1 = registrationORB_tiled(targets, sources, paras,
                                                                        write_path=write_path,
                                                                        output_type=args.outputType,
                                                                        input_type=args.inputType,
                                                                        save_target=save_target,
                                                                        imadjust=str2bool(args.imadjust))
        else:
            wrap_size, _, _ = registrationORB_tiled(targets, sources, paras,
                                                    write_path=write_path,
                                                    output_type=args.outputType,
                                                    input_type=args.inputType,
                                                    save_target=save_target,
                                                    imadjust=str2bool(args.imadjust),
                                                    keypoints1=keypoints1,
                                                    descriptors1=descriptors1)

        print("\nRegistrationORB function for round", source_round, " finished!\n====================\n")

        print("Result of " + target_round + "-" + source_round + " saved in ", write_path)
        wrap_size_min[0] = wrap_size[0] if wrap_size[0] < wrap_size_min[0] else wrap_size_min[0]
        wrap_size_min[1] = wrap_size[1] if wrap_size[1] < wrap_size_min[1] else wrap_size_min[1]

    wrap_size_check(write_path, write_path, wrap_size_min)  # make sure images are all in same size

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


#TODO: add target round as input argument --target_round=2
#TODO: add channel patterns as input argument --channel_pattern=C1.tif
#TODO: add option to register specific channels: source_round_ls
#TODO: matlab cannot read saved images... byte missmatch