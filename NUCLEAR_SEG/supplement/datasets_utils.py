"""
Utilis as additional functions

python datasets_utils.py -c ann \
--pd "/project/hnguyen/xiaoyang/exps/Data/50_plex/jj_final/seg_results/[retest]-autosegPseed-mrcnntest-imagenet-defaultBest/merged_labelmask.h5" \
--val_crops /project/hnguyen/xiaoyang/exps/Data/50_plex/jj_final/atlas/label_names.csv \
--img /project/hnguyen/xiaoyang/exps/Data/50_plex/jj_final/images_stacked/DPH.tif \
-o /project/hnguyen/xiaoyang/exps/Data/50_plex/jj_final/seg_results/[retest]-autosegPseed-mrcnntest-imagenet-defaultBest


result_dir="/project/hnguyen/xiaoyang/exps/Data/50_plex/jj_final/atlas/multiplex"
mkdir $result_dir
python datasets_utils.py -c crop \
--img /project/hnguyen/xiaoyang/exps/Data/50_plex/jj_final/images_stacked_multiplex/multiplex_adjusted.tif \
-o "$result_dir" \
--val_crops=$data_root"/atlas/label_names.csv"

Author Rebecca
"""

import sys, os
import csv
import math
import numpy as np
import skimage
from skimage import exposure, segmentation,morphology,io,img_as_ubyte,measure
from skimage.external import tifffile as tiff
import multiprocessing
from sklearn import mixture
from functools import partial
from itertools import repeat
import h5py
import pandas as pd
import glob
import scipy
import random
from sklearn.metrics import precision_recall_curve, roc_curve
import pickle
# from tqdm import tqdm
import matplotlib        
matplotlib.use('Agg')  # Agg backend runs without a display
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# Import Mask RCNN
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
def zeropadding(img, canvas_size = (100,100) ,center = False) :
    # canvas size must larger than img size
    start_i = int(canvas_size[0]/2-img.shape[0]/2) if center is True else 0
    start_j = int(canvas_size[1]/2-img.shape[1]/2) if center is True else 0
 
    if len(img.shape) ==3 : # RGB/ Multiplex
        paddedImg = np.zeros( (canvas_size[0],canvas_size[1],img.shape[2]))
        paddedImg[  start_i : start_i + img.shape[0],
                    start_j : start_j + img.shape[1],:] = img
        
    else : # 2d
        paddedImg = np.zeros((canvas_size[0],canvas_size[1]))
        paddedImg[  start_i : start_i +  img.shape[0],
                    start_j : start_j +  img.shape[1]] = img
              
    return  paddedImg       

def image_adjust( image, perc=None):
    if image.ndim ==3:   # RGB
        CHN_img_ls = []
        for i in range(image.shape[2]):
            CHN_img = image [:,:,i]
            if str(perc).isdigit() :    # use rescale_intensity
                p2, p98 = np.percentile(CHN_img, (2, perc))
                CHN_img_ls. append( exposure.rescale_intensity(CHN_img, in_range=(p2, p98)) )           # Load over images
            else:                       # equalize_adapthist
                CHN_img_adjusted = exposure.rescale_intensity(CHN_img) if CHN_img.max()> 0 else CHN_img
                CHN_img_ls. append(CHN_img_adjusted  )           # Load over images
                
                # CHN_img_ls. append( exposure.equalize_adapthist(CHN_img) )           # Load over images
        image = np.dstack (CHN_img_ls)
    else:           # gray scale
        if str(perc).isdigit() :    # use rescale_intensity
            p2, p98 = np.percentile(image, (2, perc))
            image = exposure.rescale_intensity(image, in_range=(p2, p98))            # Load over images
        else:                           # equalize_adapthist
            image = exposure.rescale_intensity(image) if image.max()> 0 else image         # Load over images
    image = skimage.img_as_ubyte(image)
    return image.copy()

def bgnoise_est(gray_image):
    ''' Estimate the bg noise threshold '''
    assert gray_image.ndim==2
    #1  prepare training set for GMM
    x_train = []
    numOfCrops =20
    crop_weights, crop_heights = [64,64]
    np.random.seed(100)
    coord_x_ls = np.random.randint( 0, gray_image.shape[0]-crop_weights, size = numOfCrops  )
    coord_y_ls = np.random.randint( 0, gray_image.shape[1]-crop_weights, size = numOfCrops  )
    for i in range(numOfCrops):
        coord_x, coord_y = coord_x_ls[i],coord_y_ls[i]
        crop = gray_image[coord_x:coord_x+crop_weights,
                                coord_y:coord_y+crop_heights]

        x_train += list( crop.reshape(-1) )                                 # contantenate the pixels in all crops 
    # 2  Run GMM clustering 
    x_train = np.array(x_train).reshape(-1, 1)
    cluster = mixture.GaussianMixture(n_components= 4, covariance_type='full',random_state=0,
                                max_iter=100).fit(x_train)
    s_order = sum( np.argsort(cluster.means_,axis=0).tolist(),[] )                              #   get the ascend order of the cluster labels
    
    label_demo = cluster.predict(x_train)    
    # 3  Designate the threshold as the middle between 2nd and 3r clustering
    thres12 =  np.array([x_train[label_demo==s_order[1]].max(),x_train[label_demo==s_order[2]].min()]).mean()  # thres is between last one and last2
    
    return thres12

def cropping (original_image, cropRange):
    if len(cropRange) == 4 :                                                                            # in the form of    (xmin, ymin, xmax, ymax), need to change  to  [[ymin, ymax],[xmin,xmax]]    
        xmin = cropRange[0]
        xmax = cropRange[2]
        ymin = cropRange[1]
        ymax = cropRange[3]
        cropRange_temp = [[xmin,xmax],[ymin, ymax]] 
        cropRange = cropRange_temp
    #			print (cropRange)            
	
    if len(original_image.shape) ==3 :
        image = original_image[cropRange[0][0]:cropRange[0][1],cropRange[1][0]:cropRange[1][1],:]    
    else:
        image = original_image[cropRange[0][0]:cropRange[0][1],cropRange[1][0]:cropRange[1][1]]   

    return image          

def crop_by_fig(image,save_fig_ls,output_dir,crop_size=[512,512] ,max_label = 500,imadjust=False):
    
    for save_fig in save_fig_ls:
        crop_x = int(save_fig.split("_")[0])        
        crop_y = int(save_fig.split("_")[1].split(".")[0]) if "." in save_fig  else int(save_fig.split("_")[1])
        crop_range     = [ crop_x, crop_y, crop_x+crop_size[0], crop_y+crop_size[1] ] # [xmin,ymin,xmax], ymax]] 
        cropped_img   = cropping(image  ,crop_range).copy()
        io.imsave (os.path.join(output_dir,save_fig + ".tif"),
                    cropped_img.astype(np.uint8))

def masks_to_label(masks):
    '''
    masks: [height, width, num_instances], binary
    label_img:  [height, width], np.int: 0~num_instances
    '''
    label_img = np.zeros([masks.shape[0],masks.shape[1]],dtype= np.int32)
    for obj_i in range(masks.shape[2]):
        mask = masks[:, :, obj_i]
        label_img[mask] = obj_i +1
    return label_img


def create_masks(bin_mask):
    """
    Create instance masks array based on binary mask
    :param bin_mask: binary mask, shape [None, None]
    :return: instance masks: shape [None, None, num_of_instances]
    """

    labeled_mask = morphology.label(bin_mask)
    labeled_mask = cleanEdgedmask(labeled_mask)

    num_instances = len(np.unique(labeled_mask)) - 1
    instance_masks = np.zeros(bin_mask.shape + (num_instances,))
    for i, c in enumerate(np.unique(labeled_mask)[1:]):
        instance_masks[labeled_mask == c, i] = 1
    return instance_masks

def cleanEdgedmask(mask, edgeWidth = 5 ):
    #'''
    #Input: mask (pixel mask array, single cell mask), label Image
    #Output: clean the componnets touching the edge 
    #''' 
    cleaned_mask = np.copy(mask)

    for obj in measure.regionprops(cleaned_mask.astype(np.int)):
        if mask.shape[0] - obj.bbox [2] < 0 or mask.shape[1] - obj.bbox [3] < 0:
            print ("EROR cleanEdgedmask!!!!!!!!")
        if obj.bbox [0] <=  edgeWidth or \
           obj.bbox [1] <=  edgeWidth or \
           mask.shape[0] - obj.bbox [2]  <=  edgeWidth or \
           mask.shape[1] - obj.bbox [3]  <=  edgeWidth : 
           cleaned_mask[cleaned_mask == obj.label] = 0
        #    print ("Edgedmask has been removed ")

    return cleaned_mask

def compute_overlaps_masks_coords(masks1_coords, masks2_coords):
    """Computes IoU overlaps between two sets of masks.
    masks1_coords,list [[x1,x2,x3...],[y1,y2,y3,y4.....]]
    masks2: [Height, Width, instances]
    """    
    # If either set of masks is empty return empty result
    if len(masks1_coords) == 0 or len(masks2_coords) == 0:
        return 0    
    
    coords1 = np.array(masks1_coords).transpose().tolist()                                          # coord list [[x1,y1],[y2,y2]....]
    coords2 = np.array(masks2_coords).transpose().tolist()
    
    # Get the intersection  and union coordinates

    intersections = [i for i in coords1 if i in coords2]    
    union = coords1.copy()
    [union.append(i) for i in coords2 if i not in coords1]
    
    # intersections and union
    return len(intersections),len(union)


def nms_check(original_label, mask_coords_abs,max_iou = 0.2):
    ''' load a the coordinate of a new object, check whether it already shown in previous original_label 
    original_label: the orignial label image  [n by m]
    mask_coords_abs: the coordinate of the new object to be checked [x<=n, y<=m]
    max_iou, the smallest iou to define two objects are the same.
    '''     
    suspicious_obj_numbers = 0
    new_obj_size = len(mask_coords_abs[0])
    # Collect the suspicious_obj_ids
    suspicious_obj_ids = []
    for coord_x ,coord_y in zip(mask_coords_abs[0],mask_coords_abs[1] ):                            # check all the pixel of new mask 
        
        suspicious_obj_id = original_label[coord_x,coord_y]                      
        if suspicious_obj_id > 0:                                                                   # if the new coord have non-zeros values in orignal mask
            if suspicious_obj_id not in suspicious_obj_ids:
                suspicious_obj_ids.append(suspicious_obj_id)
    
    # exam all the suspicious_objs 
    if len(suspicious_obj_ids) > 0:
        for obj_id in suspicious_obj_ids:
            mask_coords_existing = np.where(original_label == obj_id )
            I,U = compute_overlaps_masks_coords(mask_coords_existing, mask_coords_abs)
            IOU = I/U
            if IOU > max_iou or I/new_obj_size > max_iou:       #detect if it is a replicate object: iou> max and  subset 
                suspicious_obj_numbers = suspicious_obj_numbers + 1     
                
    # only add new label when all iou is low
    if suspicious_obj_numbers == 0:                                                              
        return True
    else:
        return False

def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    https://github.com/matterport/Mask_RCNN/blob/4c08252cc343027880a4c321213088260e732d47/mrcnn/utils.py
    """
    
    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union
    return overlaps


############################################################
#  RLE Encoding
# https://github.com/matterport/Mask_RCNN/blob/4c08252cc343027880a4c321213088260e732d47/mrcnn/utils.py
############################################################
def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))

def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask
    
def mask_to_rle(imgid, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(imgid)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)

        lines.append("{}, {}".format(imgid, rle))
    return "\n".join(lines)

def maskScoreClass_to_rle(imgid, mask, scores,class_ids):

    '''
    imgid: image name:
    mask:binary masks : Mask must be [H, W, count]
    scores: [count]
    '''

    "Encodes instance masks to submission format."

    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(imgid)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue        
        rle = rle_encode(m)
        lines.append("{}, {},{},{}".format(imgid, rle,scores[o-1], class_ids[o-1]))
    return "\n".join(lines)

def mask_bg(masks):
    # input:
    #    masks: [height, width, num_instances]
    # Output:
    #    The backgourd binmask that never visited  [0: bg, 1: masked]
    bg = np.zeros(masks.shape[:2])
    for i in range(masks.shape[2]):
        bg[masks[:,:,i] > 0] = 1 
    return bg


############################################################
#  Feature table  for detection
############################################################

############################################################
#  Feature table  for detection
############################################################

## only for ftable function
def generate_featureTable_allCHN( label_img, image, CHN_dic= {0:"D",1:"H",2:"DPH",3:"NeuN",4: "IBA1",5:"Olig2",6:"S100",7:"RECA1"} ):
    if type(image) == dict:
        CHN_nums = len(image)
    else:
        if len(image.shape) ==2: # gray scale
            image = np.dstack([image])
        CHN_nums = image.shape[2]   
        
    ls_dict = {}        
    
    for obj_id , obj in enumerate( measure.regionprops(label_img)):       
        if obj_id ==0:
            ls_dict['ID'          ]    = []
            ls_dict['centroid_x'          ]= []
            ls_dict['centroid_y'          ]= []
            ls_dict['xmin'             ]= []
            ls_dict['ymin'             ]= []
            ls_dict['xmax'             ]= []
            ls_dict['ymax'             ]= []                                                     

            #direct properties                                  
            ls_dict['area'               ]= []
            ls_dict['bbox_area'          ]= []
            ls_dict['convex_area'        ]= []
            ls_dict['filled_area'        ]= []
            ls_dict['eccentricity'       ]= []
            ls_dict['equivalent_diameter']= []
            ls_dict['orientation'        ]= []
            ls_dict['perimeter'          ]= []
            ls_dict['solidity'           ]= []
            # ls_dict['Elongation'         ]= []
            for CHN_id in range(CHN_nums):           # the CHN might be smaller than CHN_dic.keys
                ls_dict[CHN_dic[CHN_id] +'__weighted_centroid_x']=[]
                ls_dict[CHN_dic[CHN_id] +'__weighted_centroid_y']=[]                                                                                  
                ls_dict[CHN_dic[CHN_id] +'__Sum'               ] =[]                                                                                  
                ls_dict[CHN_dic[CHN_id] +'__Mean'               ]=[]
                ls_dict[CHN_dic[CHN_id] +'__Max'                ]=[]
                ls_dict[CHN_dic[CHN_id] +'__Area'                ]=[]
                ls_dict[CHN_dic[CHN_id] +'__Variance'           ]=[]
                ls_dict[CHN_dic[CHN_id] +'__Entropy'            ]=[]
                ls_dict[CHN_dic[CHN_id] +'__Skew'               ]=[]
                ls_dict[CHN_dic[CHN_id] +'__Centroid_Shift'] = []
                ls_dict[CHN_dic[CHN_id] +'__AreaPerc'] = []                   

        # lamda_1           = obj.major_axis_length
        # lamda_0           = obj.minor_axis_length
        # Elongation        = 0 if lamda_0 == 0 else lamda_1/lamda_0  
        ls_dict['ID'               ].append( obj.label       )
        ls_dict['centroid_x'          ].append( int(obj.centroid[0] )       )
        ls_dict['centroid_y'          ].append( int(obj.centroid[1] )        )
        ls_dict['xmin'             ].append( obj.bbox[0]                     )
        ls_dict['ymin'             ].append( obj.bbox[1]                     )
        ls_dict['xmax'             ].append( obj.bbox[2]                     )
        ls_dict['ymax'             ].append( obj.bbox[3]                     )                                                     

        #direct properties                                  
        ls_dict['area'               ].append( obj.area                     )
        ls_dict['bbox_area'          ].append( obj.bbox_area                )
        ls_dict['convex_area'        ].append( obj.convex_area              )
        ls_dict['filled_area'        ].append( obj.filled_area              )
        ls_dict['eccentricity'       ].append( obj.eccentricity             )
        ls_dict['equivalent_diameter'].append( obj.equivalent_diameter      )
        ls_dict['orientation'        ].append( obj.orientation              )
        ls_dict['perimeter'          ].append( obj.perimeter                )
        ls_dict['solidity'           ].append( obj.solidity                 )
        # ls_dict['Elongation'         ].append( Elongation                   )        
        for CHN_id in range(CHN_nums):
            if type(image) == dict:
                CHN_img = image[CHN_id][obj.bbox[0]:obj.bbox[2],
                                        obj.bbox[1]:obj.bbox[3]].copy()
            else:
                CHN_img = image[obj.bbox[0]:obj.bbox[2],
                            obj.bbox[1]:obj.bbox[3],CHN_id]
            CHN_img = CHN_img*(obj.filled_image)  # remove the background

            if CHN_img.sum()==0:                                                                # if no pixel in CHN, assign the basic center
                weighted_centroid_global = obj.centroid
            else:                
                CHN_objs = measure.regionprops(obj.filled_image*1,CHN_img )
                weighted_centroid_global = CHN_objs[0].weighted_centroid  + np.array([obj.bbox[0],obj.bbox[1]])

            centroid_shift =   scipy.spatial.distance.euclidean ( weighted_centroid_global,obj.centroid)
            pdf = CHN_img.reshape(-1)
            #secondary properties                              
            ls_dict[CHN_dic[CHN_id] +'__weighted_centroid_x'].append( weighted_centroid_global[0]  )
            ls_dict[CHN_dic[CHN_id] +'__weighted_centroid_y'].append( weighted_centroid_global[1]  )                                                                                  
            ls_dict[CHN_dic[CHN_id] +'__Sum'               ].append( pdf.sum()                    )                                                                                  
            ls_dict[CHN_dic[CHN_id] +'__Mean'               ].append( pdf.sum()/obj.area           )
            ls_dict[CHN_dic[CHN_id] +'__Area'               ].append( (pdf>0).sum()                )
            ls_dict[CHN_dic[CHN_id] +'__Max'                ].append( pdf.max()                    )
            ls_dict[CHN_dic[CHN_id] +'__Variance'           ].append( scipy.stats.variation(pdf)   )
            ls_dict[CHN_dic[CHN_id] +'__Entropy'            ].append( scipy.stats.entropy(pdf)     )
            ls_dict[CHN_dic[CHN_id] +'__Skew'               ].append( scipy.stats.skew(pdf)        )
            ls_dict[CHN_dic[CHN_id] +'__Centroid_Shift'].append(centroid_shift  )
            ls_dict[CHN_dic[CHN_id] +'__AreaPerc'].append((pdf>0).sum()/obj.area  )

    return pd.DataFrame.from_dict(ls_dict)

def enlarge_bbox(obj_bbox,img_shape, enlarge_width = 10 ):
    # enlarge the bbox area    
    min_row = max( 0, obj_bbox[0] - enlarge_width )
    min_col = max( 0, obj_bbox[1] - enlarge_width )
    max_row = min(obj_bbox[2] + enlarge_width, img_shape[0] )
    max_col = min(obj_bbox[3] + enlarge_width, img_shape[1] )
    return [min_row,min_col,max_row,max_col]
##############################
# extract annotation crops
##############################

def label_savevis(img, label,img_id,output_dir,imadjust=True, max_label = 10000, class_table_crop= None):
    
    if img.ndim > 2:   # use the 3rd channel for display
        img = img[:,:,2]
    if img.ndim == 2:   # use the last channel for display
        img = skimage.color.gray2rgb(img)
    if imadjust :
        img = image_adjust(img)            

    # save visalization
    img_id = img_id.split(".")[0] if "." in img_id else img_id

    merged_visual = img.copy()
    border_abs = np.where(segmentation.find_boundaries(label))
    merged_visual [border_abs[0],border_abs[1],:] =   [0,255,0]             # RGB
    io.imsave (os.path.join(output_dir,img_id + "_vis.tif"),
                    merged_visual.astype(np.uint8))

    # save relabeled annotations and masks 
    relabel,fw,__ = segmentation.relabel_sequential (label)               # relabel to 0

    if max_label <max_label:                                             # limited color to paint
        relabel = relabel% (max_label-1) + 1                              # get the remainder of absolute label, becasue annotation software could have at most max_label colors
        relabel[label==0] = 0                                             # set background to 0   

    if class_table_crop is not None:

        class_table_crop["local_id"] = np.unique(relabel)[1:]

    io.imsave (os.path.join(output_dir,img_id + "_color.png"),
                skimage.img_as_ubyte( skimage.color.label2rgb( relabel,bg_label=0)))

    relabel= np.dstack([relabel]*3).astype(np.uint8)
    io.imsave (os.path.join(output_dir,img_id + "_mask.png"),
                relabel)    
    # save raw image
    io.imsave (os.path.join(output_dir,img_id + ".jpeg"),
                    img.astype(np.uint8))

    return class_table_crop

def bgBoost_savevis(img, labels,fglabels,bglabels,img_id,output_dir,imadjust=True):
    '''
    Visualize the bg boosting result in the middle of iterations
    bglabels: gray color
    fglabels: pseudo colors
    '''

    if img.ndim > 2:   # use the 3rd channel for display
        img = img[:,:,2]
    if img.ndim == 2:   # use the last channel for display
        img = skimage.color.gray2rgb(img)
    if imadjust :
        img = image_adjust(img)            

    # save raw image
    io.imsave (os.path.join(output_dir,img_id + ".jpeg"),
                    img.astype(np.uint8))
    # save visalization
    img_id = img_id.split(".")[0] if "." in img_id else img_id
    merged_visual = img.copy()
    border_abs = np.where(segmentation.find_boundaries(labels))
    merged_visual [border_abs[0],border_abs[1],:] =   [0,255,0]             # RGB
    io.imsave (os.path.join(output_dir,img_id + "_vis.tif"),
                    merged_visual.astype(np.uint8))

    # save relabeled annotations and masks 
    bgBoost_vis = np.zeros_like(merged_visual,dtype= np.uint8)
    bg_labels_coords_x,  bg_labels_coords_y =  np.where(bglabels>0)
    bgBoost_vis[bg_labels_coords_x,bg_labels_coords_y] =  [128,128,128]     # set bg objects to gray
    bgBoost_vis[fglabels > 0] = 0                                           # clear the fg to 0
    fg_colored = skimage.color.label2rgb( fglabels,bg_label=0)*255
    bgBoost_vis += fg_colored.astype(np.uint8)                                              # set the fg objs to colors
    io.imsave (os.path.join(output_dir,img_id + "_bgBoost_vis.png"),
                skimage.img_as_ubyte( bgBoost_vis))



def annotation(label,image,save_fig_ls,output_dir,crop_size=[512,512] ,
                max_label = 500,imadjust=True,class_table = None):
    # annoation:
    # max_label to small number , relabel to 0~11 for manual correction
    # max_label to the largest possible number of object: for validation

    class_table_out,ctable_crop,ctable = None,None,None

    if class_table is not None:
        ctable = pd.read_csv(class_table)


    for save_fig in save_fig_ls:
        if type(label)==np.ndarray:
            crop_x = int(save_fig.split("_")[0])        
            crop_y = int(save_fig.split("_")[1].split(".")[0]) if "." in save_fig  else int(save_fig.split("_")[1])
            crop_range     = [ crop_x, crop_y, crop_x+crop_size[0], crop_y+crop_size[1] ] # [xmin,ymin,xmax], ymax]] 
            crop_label    = cropping(label  ,crop_range).copy()  
            cropped_img   = cropping(image  ,crop_range).copy()

            if ctable is not None:
                ctable_crop = ctable.loc[list(np.unique(crop_label)[1:])]
                ctable_crop["atlas_crop"] = [save_fig.split(".")[0]]*len(ctable_crop)
                ctable_crop["local_id"]  = [0] * len(ctable_crop)

            if cropped_img.sum()>0:                                                         # crop_range have detected 
                ctable_crop = label_savevis(img = cropped_img, label = crop_label,
                              img_id = save_fig, output_dir = output_dir,
                              imadjust=imadjust, max_label = max_label,class_table_crop = ctable_crop )

            if ctable is not None and ctable_crop is not None:
                if class_table_out is None:
                    class_table_out = ctable_crop
                else:
                    class_table_out = class_table_out.append(ctable_crop)
                # import pdb; pdb.set_trace()

    if class_table_out is not None:
        class_table_out.to_csv(os.path.join(output_dir,"atlas_" + os.path.basename(class_table)))


def mask2props(mask, RemoveEdge = True):
    '''
    input:
    @mask:annotation color masks with color map (the max number might not be the number of objs)
    outputs:
    @paras:boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    @masks: [height, width, num_instances] binmary masks
    '''
    label = measure.label (mask)        # separate all the unconnected labels
    if RemoveEdge:
        label = cleanEdgedmask(label ,edgeWidth= 2)
    label,__,__ = segmentation.relabel_sequential (label)
    # label = morphology.remove_small_objects(label, 100)
    objs_dict = measure.regionprops(label)   
    boxes_ls = []
    binmask_ls = []
    for obj in objs_dict:                    
        boxes_ls.append(obj.bbox)
        binmask_ls.append(label == obj.label)
    
    class_id = np.ones(len(objs_dict))
    return np.array(boxes_ls), class_id, np.stack(binmask_ls,axis =2),label


def compute_matches_bymasks(pd_mask,gt_mask, iou_threshold = 0):
    '''
    Replacement of compute the match without pd score
    By Pengyu
    '''

    overlaps = compute_overlaps_masks(pd_mask, gt_mask)  # shape [87, 90]
    if len(overlaps) == 0:
        print("There is no match in ")
        return None
    ids = np.argsort(np.max(overlaps, axis=-1))
    overlaps = overlaps[ids[::-1]]                
    # Loop through predictions and find matching ground truth boxes
    
    match_count = 0
    pd_match = -1 * np.ones([overlaps.shape[0]])
    gt_match = -1 * np.ones([overlaps.shape[1]])
    for i in range(len(overlaps)):
        # Find best matching ground truth box
        # 1. Sort matches by iou
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low iou
        low_score_idx = np.where(overlaps[i, sorted_ixs] <= iou_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # We have a match
            else:
                match_count += 1
                gt_match[j] = i
                pd_match[i] = j
                break
    return gt_match, pd_match,overlaps



def str2bool(str_input):
    bool_result = True if str_input.lower() in ["t",'true','1',"yes",'y'] else False
    return bool_result



if __name__ == '__main__':
    import argparse,time
    import matplotlib
#     # Agg backend runs without a display
    matplotlib.use('Agg')
    tic = time.time()   
    # Parse command line arguments

    parser = argparse.ArgumentParser(
        description='Calculate IOU over 2 whole brain label mask')

    parser.add_argument("--command","-c",
                        metavar="<command['iou','ftable','iou_ann']>", default = 'iou',
                        help="'iou', 'iou_ann' or 'generate_feature table' or 'generate annotation table'")
    parser.add_argument('--output_dir',"-o", required= False,
                        metavar = "/path/to/maskfile/",
                        default = 'iou_df.csv',
                        help='Full name to save as result ')                          
    parser.add_argument('--pd', required=True,
                        metavar="/path/to/dataset/",
                        default = "/uhpc/roysam/xiaoyang/exps/NuclearSeg_DNN/Mask_RCNN/datasets/CHN50/[DAPI+Histones]208835seeds_Labels.out",
                        help='pd_labels txt')   
    parser.add_argument('-a','--imadjust', required=False,
                        default = '0',type = str, 
                        help='whether to adjust the image')                                                   
    parser.add_argument('--ctable', required= False,
                        metavar = "/classtable/",
                        default = None,
                        help='classtable from whole image')                                   
    args, _ = parser.parse_known_args()


    if args.command == "ftable" :
        parser.add_argument('--img', required=True,
                        metavar="/path/to/dataset/",    default = None,
                        help='Whole image to apply cell segmentation')
    elif args.command == "crop" or "ann":
        parser.add_argument('--img', required=True,
                        metavar="/path/to/dataset/",    default = None,
                        help='Whole image to apply cell segmentation')
        parser.add_argument('--val_crops', required=False,default="/project/ece/roysam/xiaoyang/exps/Data/50_plex/jj_final/atlas/label_names.csv",
                        metavar="/path/to/crop_img_id.csv/",
                        help='the path to load the crop_img id for savinng the detection result for validation')   

    args = parser.parse_args()    

    ###   running
    os.makedirs(args.output_dir,exist_ok=True)
 
    if args.command == "ftable":
        hf = h5py.File(args.pd, 'r')                                                             # load wholelabel use 9s
        pd_labels = np.array(hf.get('seg_results'))
        hf.close()
        image = tiff.imread (args.img)        
        print ("img and label loaded....")
        featureTable_df =  generate_featureTable_allCHN(pd_labels,image)  # image is better to be multiplex (8chn)
        featureTable_df.to_csv (args.output_dir)

        # clumps_df = clump_detection(pd_labels)
        # clumps_df.to_csv (args.output_dir.split(".csv")[0] + "_clumps.csv")
        print("save featureTable_df to ",args.output_dir)

    elif args.command == "crop":
        with tiff.TiffFile(args.img) as tif:
            wholeImage = tif.asarray(memmap=True)
        
        df_names = pd.read_csv(args.val_crops)
        save_fig = df_names["crop_img"]. tolist()
        output_dir = os.path.join( args.output_dir , "multiplex_atlas") 
        crop_by_fig(image = wholeImage ,
                    save_fig_ls = save_fig,
                    output_dir = output_dir,
                    imadjust = str2bool(args.imadjust))

    elif args.command == "ann":  # create ann pd images
        hf = h5py.File(args.pd, 'r')                                                             # load wholelabel use 9s
        pd_labels = np.array(hf.get('seg_results'))
        hf.close()
        with tiff.TiffFile(args.img) as tif:
            wholeImage = tif.asarray(memmap=True)

        if wholeImage.ndim >2:
            wholeImage = wholeImage[:,:,2].copy()
        df_names = pd.read_csv(args.val_crops)
        save_fig = df_names["crop_img"]. tolist()

        output_dir = os.path.join( args.output_dir , "pred_atlas") 
        os.makedirs( output_dir,exist_ok= True)
        annotation(label = pd_labels, 
                    image = wholeImage ,
                    save_fig_ls = save_fig,
                    output_dir = output_dir,
                    imadjust = str2bool(args.imadjust),
                    class_table = args.ctable)

    toc = time.time()
    print ("save results at:",args.output_dir)

    print ("Time = ", str(toc-tic) ,"\t\t\t\t")



