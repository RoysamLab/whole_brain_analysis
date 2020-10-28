#!/usr/bin/env python
# coding: utf-8
'''
conda activate seg

usr_root="/project/hnguyen/xiaoyang"
data_root=$usr_root"/exps/Data/50_plex/jj_final"

seg_dir=$data_root/seg_results/[retest]-autosegPseed-mrcnntest-imagenet-defaultBest

cd $usr_root/exps/SegmentationPipeline/mrcnn_Seg/supplement

python split_celltypeMasks.py \
--label=$seg_dir/"merged_labelmask.h5" \
--fTable_LOC=$seg_dir/"fTable_merged.csv" \
--img_LOC=$data_root/"images_stacked_multiplex/multiplex.tif" \
--val_crops=$data_root/atlas/label_names.csv \
--mp=True \
--imadjust=False \
--output_dir=$seg_dir/"SegAnomoly"

python split_celltypeMasks.py \
--label=$seg_dir/"merged_labelmask.h5" \
--fTable_LOC=$seg_dir/"fTable_merged.csv" \
--img_LOC=$data_root/"images_stacked_multiplex/multiplex.tif" \
--val_crops=$data_root/atlas/label_names.csv \
--mp=True \
--imadjust=True \
--output_dir=$seg_dir/"SegAnomoly_adjust"

'''

# In[1]:
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os, sys
import numpy as np
import skimage
from skimage.external import tifffile as tiff
from skimage import exposure, segmentation,io,img_as_ubyte,measure,morphology
# from skimage.segmentation import find_boundaries
import pandas as pd
import multiprocessing
from itertools import repeat
import random 
from sklearn import mixture ,linear_model
from sklearn.mixture import GaussianMixture
import time
import pandas as pd
import h5py
import argparse
from scipy import ndimage as ndi
from scipy import stats
import warnings

ProjRoot = "/project/ece/roysam/xiaoyang/exps/SegmentationPipeline"
sys.path.insert(0, os.path.join(ProjRoot,"mrcnn_Seg/supplement"))
import datasets_utils as dt_utils

sys.path.insert(0, os.path.join(ProjRoot,"Automatic_Seg/sub_fcts"))
import CellSegmentationfcts as seg_fcts
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(
    description='Run Cell Phenotyping and Cell split check at the same time')

parser.add_argument('-l',"--label", required=False,
                    default="/project/ece/roysam/xiaoyang/exps/Data/50_plex/jj_final/seg_results/[retest]-autoseg-mrcnn1-bb-split-mrcnn2-bb/merged_labelmask.h5",
                    metavar="/path/to/label.h5/",
                    help='the path to load merged label image')   
parser.add_argument('-i','--img_LOC', required=False,
                    metavar = "/path/to/multiplex.tif/",
                    default = "/project/ece/roysam/xiaoyang/exps/Data/50_plex/jj_final/images_stacked_multiplex/multiplex.tif",
                    help='multiplex.tif')
parser.add_argument('-f','--fTable_LOC', required=False,type=str,
                    metavar ="/path/to/fTable_merged.csv/",
                    default ="/project/ece/roysam/xiaoyang/exps/Data/50_plex/jj_final/seg_results/[retest]-autoseg-mrcnn1-bb-split-mrcnn2-bb/fTable_merged.csv",
                    help='csv file contain   ID, xmin,ymin,xmax,ymax, ID correspond to label')
parser.add_argument('--output_dir',"-o", required= False,
                    metavar = "/path/to/maskfile/",
                    default = "/project/ece/roysam/xiaoyang/exps/Data/50_plex/jj_final/seg_results/[retest]-autoseg-mrcnn1-bb-split-mrcnn2-bb/split_cells",
                    help='Cell spliting results')                          
parser.add_argument('--thres',"-t", required= False,
                    metavar = "threshold of coefficient,recomend:=0.4,force FP:=0",
                    default = 0.2, help="the lower threshold of sparse coefficent of positive cell type, the smaller this value is , the more splitted cells ")                          
parser.add_argument('--mp', required=False, type=str,
                    default ="True",
                    help='multiprocessing')   
parser.add_argument('--imadjust', required=False,
                    default ="False",type = str, 
                    help='whether to adjust the image')                                                            
parser.add_argument('--val_crops', required=False,
                    default="/project/ece/roysam/xiaoyang/exps/Data/50_plex/jj_final/atlas/label_names.csv",
                    metavar="/path/to/crop_img_id.csv/",
                    help='the path to load the crop_img id for training the pixel clustering model')   
parser.add_argument('--vis', required=False,
                    default="0",
                    help='whether to save the vis results')
parser.add_argument('--remove_lowconfidence', required=False, type=str,
                    default ="False",
                    help='whether or not remove low confidence (FP) objects')   
args, _ = parser.parse_known_args()
Verbose= dt_utils.str2bool(args.vis)
remove_lc = dt_utils.str2bool(args.remove_lowconfidence)
print("Verbose=",Verbose,"remove_lc=",remove_lc)

#%%
t0 = time.time()
print(''' \n1. load nuclear mask and feature table ''')
write_path = args.output_dir
os.makedirs(write_path,exist_ok=True)
write_objSplit_path = os.path.join(write_path,"split_cells")
os.makedirs(write_objSplit_path,exist_ok=True)
write_objCorrect_path = os.path.join(write_path,"correct_cells")
os.makedirs(write_objCorrect_path,exist_ok=True)

hf = h5py.File(args.label, 'r')                                                             # load wholelabel use 9s
wholelabel = np.array(hf.get('seg_results'))
hf.close()

objs_dict = pd.read_csv(args.fTable_LOC)                                                    # load the table 
new_dict = objs_dict[["ID","centroid_x","centroid_y","xmin","ymin","xmax","ymax"]]          # new dict for saving the splited featureTable in the end
objs_dict = objs_dict.set_index("ID")
# multiplex =  tiff.imread(args.img_LOC)

with tiff.TiffFile(args.img_LOC) as tif:
    multiplex = tif.asarray(memmap=True)

whole_borders = skimage.color.rgb2gray( tiff.imread( os.path.join( os.path.dirname(args.label),
                                                                  "merged_result_borders.tif" ))
                                        ).astype(np.bool)

CTypeNames_dic = {"NeuN":3, "IBA1":4,"Olig2":5,"S100":6,"GFP":7}                          # cell type names and their channel id at multiplex
CTypemultplex = ["DAPI","Histone","DPH","NeuN", "IBA1","Olig2","S100","GFP"]


if dt_utils.str2bool(args.imadjust):
    print ("Applying img adjust")
    multiplex = dt_utils.image_adjust(multiplex)



#%%
t1 = time.time()
print("Used time=", int(t1-t0))
print(''' \n2. Create cell type masks by thresholding''')

# 1. Load Inputs
multiplex_celltype_mask = np.zeros_like( multiplex, dtype= np.bool)
multiplex_celltype_mask[:,:,:3] = 1

# only interest in the pixels in side the label mask
multiplex = multiplex * np.dstack ([wholelabel>0]*multiplex.shape[2])                      # foreground  = intersection( nuclear mask ,cell type)

CTypethres_dic = {}                                                                         # the thresholds for later use
method = "GMM"                                                                              # Method to do pixel clustering
demo = True
if demo:   # use predifined threshold to save time


    CTypeParas_dic= {'NeuN': {'n_components': 3, 'thres_ratio': 1.3, 'min_size': 120, 'c': [0, 255, 0], 'thres': 14.950000000000001}, 
                    'IBA1': {'n_components': 4, 'thres_ratio': 1.05, 'min_size': 80, 'c': [255, 0, 0], 'thres': 25.725}, 
                    'Olig2': {'n_components': 4, 'thres_ratio': 0.95, 'min_size': 90, 'c': [128, 0, 128], 'thres': 14.725}, 
                    'S100': {'n_components': 4, 'thres_ratio': 1, 'min_size': 150, 'c': [255, 165, 0], 'thres': 24.5}, 
                    'GFP': {'n_components': 5, 'thres_ratio': 0.7, 'min_size': 100, 'c': [255, 255, 0], 'thres': 21.349999999999998}}
else:   # generaten threshold from image 

    CTypeParas_dic = {"NeuN":  {"n_components":2,"thres_ratio":1.1 ,"min_size":120, "c":[0,255,0]},  # geen
                     "IBA1":  {"n_components":2,"thres_ratio":1,"min_size":80,  "c":[255,0,0]},# Red
                     "Olig2": {"n_components":2,"thres_ratio":1,"min_size":90,  "c":[128,0,128]},  #purple
                     "S100":  {"n_components":2,"thres_ratio":1   ,"min_size":150, "c":[255,165,0]}, # orange
                     "GFP":   {"n_components":2,"thres_ratio":1 ,"min_size":100, "c":[255,255,0]}}

    # 2 Train 3 class pixel-based clustering 
    ## 2.1 Prepare Training set: extract crops from atlas ##'''Use a small subset of image to train the clustering model globally'''
    coord_x_ls, coord_y_ls = [],[]
    crop_weights, crop_heights = [20,20]
    x_train_dict = {}
    if args.val_crops is not None:              # extract the coord ids by atlas
        atlas_names = pd.read_csv(args.val_crops)
        for crop_coords in atlas_names["crop_img"]:
            coord_x = int( crop_coords.split("_")[0]) + 100                                         # experiment: crops locate at the center 
            coord_y = int( crop_coords.split("_")[1]) + 100
            coord_x_ls.append(coord_x)
            coord_y_ls.append(coord_y)
    else:                                        # 200 random coords
        coord_x_ls = np.random.random_integers( 0 , multiplex.shape[0] - 100 , 200 )
        coord_y_ls = np.random.random_integers( 0 , multiplex.shape[1] - 100 , 200 )

    for coord_x,coord_y in zip(coord_x_ls, coord_y_ls ):
        multiplex_crop = multiplex[coord_x:coord_x+crop_weights,
                                   coord_y:coord_y+crop_heights,:]

        for  c_name in  CTypeNames_dic:
            if c_name not in x_train_dict.keys():
                x_train_dict[c_name] = []            
            chn_crop = multiplex_crop[:,:, CTypeNames_dic[c_name]]
            x_train_dict[c_name] += list( chn_crop.reshape(-1))                                 # contantenate the pixels in all crops 
    nuclear_label_CHN = {}

    # 2.2 Train 3 class pixel-based clustering 
    for  c_name in  list(CTypeNames_dic.keys()):
        CHN_id = CTypeNames_dic[c_name]
        data = np.array(x_train_dict[c_name])
    #     data = np.log(data[data>0])
        x_train = np.array(data).reshape(-1,1)

        # print (c_name, "X_Train:", len(x_train))
        if method =="GMM":        
            n_components = CTypeParas_dic[c_name]["n_components"]
            cluster = GaussianMixture(n_components= n_components, covariance_type='full',random_state=0,
                                    max_iter=100).fit(x_train)

        s_order = sum( np.argsort(cluster.means_,axis=0).tolist(),[] )                              #   get the ascend order of the cluster labels
        label_demo = cluster.predict(x_train)    
        thres12 =  np.array([x_train[label_demo==s_order[-2]].max(),x_train[label_demo==s_order[-1]].min()]).mean()  # thres is between last one and last2
    #     thres12 = np.exp(thres12)
        thres12 = thres12 * CTypeParas_dic[c_name]["thres_ratio"]
        CTypeParas_dic[c_name]["thres"] = thres12   # add the calucalted threshold value
        # print (c_name,": thres=",thres12)

        CTypethres_dic [c_name] = thres12
        if args.vis :  # vis
            intensity_CHN_demo = multiplex[20000:20500,20050:20700,CHN_id]       
    #         nuclear_label_CHN = cluster.predict(intensity_CHN_demo.reshape(-1,1)).reshape(intensity_CHN_demo.shape[:2])
            nuclear_label_CHN[c_name] = (intensity_CHN_demo>thres12)
            cleaned = morphology.remove_small_objects(nuclear_label_CHN[c_name] , 
                                                      min_size=CTypeParas_dic[c_name]["min_size"])
            # save fig to compare raw image and seg result for a small crop
            plt.figure()
            ax1=plt.subplot(1,2,1)
            ax1.imshow(intensity_CHN_demo,cmap="gray")
            ax1.axis('off')
            ax2 = plt.subplot(1,2,2, sharex=ax1)
            ax2.imshow(cleaned)
            ax2.axis('off')
            plt.tight_layout()
            

        if args.vis:
            # Save the cell type segmentation borders for whole brain
            intensity_CHN = multiplex[:,:,CHN_id]     # thres between the max 2 labels, avg of the pixel 
            CHN_mask = np.array(intensity_CHN>thres12)
            CHN_mask = morphology.remove_small_objects(CHN_mask, 
                                                    min_size=CTypeParas_dic[c_name]["min_size"])

            border_merged = skimage.segmentation.find_boundaries(CHN_mask)
            tiff.imsave(os.path.join( write_path , c_name+"_NuclearBorder.tif"),skimage.img_as_ubyte(border_merged*255))
            
            multiplex_celltype_mask[:,:,CHN_id ] = CHN_mask
            # multiplex_celltype_masked[:,:,CHN_id] = multiplex[:,:,CHN_id] * CHN_mask

    # tiff.imsave(os.path.join( write_path , "Multiplex_celltype_mask.tif"),skimage.img_as_ubyte(multiplex_celltype_mask))
    print ("CTypeParas_dic=",CTypeParas_dic)

t2 = time.time()
print("Used time=", int(t2-t1))

#%%
print(''' \n3. Cell type classification and cell splitting''')

def iandu_mask(CHNmask_ls,cell_mask):
    '''
    # Input
    @CHNmask_ls: the list of cell type channel binmasks
    @cell_mask: the original nuclear mask
    
    # Output
    @intersection, union
    ''' 
    cell_mask = cell_mask.astype(np.bool)*1
    tol_intersections = 0
    tol_union = 0
    for i1, mask1 in enumerate( CHNmask_ls):    
        mask1 = mask1.astype(np.bool)*1
        for i2, mask2 in enumerate( CHNmask_ls):    
            mask2 = mask2.astype(np.bool)*1
            if i1 != i2 and i2 > i1:
                m1 = mask1*cell_mask
                m2 = mask2*cell_mask

                intersections = m1*m2
                union = m1.sum() +  m2.sum()- intersections.sum ()
                tol_intersections += intersections.sum ()
                tol_union += union.sum()
            
    return tol_intersections/cell_mask.sum(),tol_union/cell_mask.sum()

def stack_newcenters(CHNmask_ls, mask_crop ,crop_img= None ):
    '''
    # Input
    @new_centers_masks: the list of cell type channel binmasks [ ] 
    @crop_mask: the original nuclear mask
    @crop_img: the watershed input image: intensity image or distance map(if None)

    # Output
    @labels: new label masks ,label correspond to binmasks ids(start from 1)
    '''
    
    labels=np.zeros_like(mask_crop,dtype=np.int)
    new_mask_sz =[m.sum() for m in CHNmask_ls]
    for center_i,new_mask_i in enumerate (np.argsort(new_mask_sz)[::-1]):
        print (center_i,new_mask_i)
        labels[np.where(CHNmask_ls[new_mask_i])   ]=center_i +1
    return labels

def watershed_newcenters(new_centers_masks, crop_mask ,crop_img= None ):
    '''
    # Input
    @new_centers_masks: the list of cell type channel binmasks [ ] 
    @crop_mask: the original nuclear mask
    @crop_img: the watershed input image: intensity image or distance map(if None)

    # Output
    @labels: new label masks ,label correspond to binmasks ids(start from 1)
    '''
    
    # prepare seeds Step 1: remove overlaps
    overlap_mask = np.zeros_like(new_centers_masks[0],dtype=np.int)

    for center_i,new_centers_mask in enumerate (new_centers_masks):
        new_centers_mask_bin = new_centers_mask>0
        new_centers_mask_bin = morphology.binary_erosion (new_centers_mask_bin,morphology.disk(2))   
        overlap_mask += new_centers_mask_bin*1 
        new_centers_masks[center_i] = new_centers_mask_bin*1
        
    overlap_mask = (overlap_mask> 1)    
    seed = np.zeros_like (overlap_mask,dtype= np.int)
    distanceMap = np.zeros_like (seed,dtype= np.float)
    
    for center_i,new_centers_mask in enumerate (new_centers_masks):
        new_centers_mask[overlap_mask] = 0  # remove overlap regions  
        masks_labels = measure.label ( new_centers_mask>0)         
        #   prepare seeds Step 2:remove small pieces:
        if masks_labels.max() > 1:
            pieces_dict = measure.regionprops_table(masks_labels, properties = ('label','area'))
            largest_id = np.argmax( pieces_dict["area"])
            new_centers_mask = ( masks_labels==pieces_dict["label"][largest_id] )       # binmask of largest component

        new_centers_mask = (new_centers_mask>0) * ( center_i +1 )                       #[binmask]*1, [binmask]*2
        # prepare seeds Step 3:remove small pieces:
        seed = seed + new_centers_mask         

    if crop_img is None:        
        # prepare distant map
        distanceMap = distanceMap + ndi.distance_transform_edt(new_centers_mask)        
   
        labels  = skimage.morphology.watershed(-distanceMap, seed, mask=crop_mask)
    else:        
        labels  = skimage.morphology.watershed(-crop_img, seed, mask=crop_mask)

    return labels

def check_splited_label(splited_label, min_perc = 0.3 ):
    ''' 
    True: more than one cell
    False: one cell (no need to split)              
    '''
    result = False
    if len( np.unique(splited_label).nonzero()[0]) > 1 :                                # criteria1: #  more than one element                
        object_size = [ (splited_label == l ).sum() for  l in np.unique(splited_label).nonzero()[0]  ]
        object_size = object_size / ( (splited_label>0).sum())
        if object_size.min() > 0.3 :                                                    # criteria1: #  smallest area of elemment size >30% of whole large
            result = True
    return result

def prepareOMP(multiplex_crop,mask_crop,CTypeParas_dic):
    '''
    Outputs
    @ X :  n_components (reshaped binary mask vector length) x n_features( number of cell types) 
    @ y :  reshaped binary mask vector
    @ min_perc : min_percentarge of cell type binary masks
    @ cell_type_channels_masked: cell type channel multiplex image  ( width x height x n_features  )
    @ CTypemask_crop:  cell type binary masks  ( width x height x n_features  )
    @ perc_mask:  cell type binary masks occupancy to mask_crop
    @ cell_type_pixel_sum: summation of cell type pixels within nuclear mask 1 x n_features( number of cell types)   (for finding unknown class)
    '''
    
    CTypemask_crop = np.zeros_like(multiplex_crop,dtype=np.int)                                            # 8CHN get the cell type mask by cell type threshold
    cell_type_pixel_sum = []
    for  c_name in CTypeNames_dic:
        CHN_id = CTypeNames_dic[c_name]
        # train_crop = multiplex_view[:,:,CHN_id]
        test_crop =  multiplex_crop[:,:,CHN_id]
        # mask_test = local_thres(train_crop,test_crop)
        global_thres = CTypeParas_dic[c_name]["thres"] 
        CHN_mask = np.array(test_crop >  global_thres)
        CHN_mask = morphology.remove_small_objects(CHN_mask, 
                                                  min_size=CTypeParas_dic[c_name]["min_size"])
        CTypemask_crop[:,:,CHN_id] = CHN_mask
        cell_type_pixel_sum .append ( np.sum( test_crop*CHN_mask * mask_crop)  )          # valid pixel intensity in each individual channel
    multiplex_masked = CTypemask_crop * np.dstack ([mask_crop]*multiplex_crop.shape[2])   # foreground  = intersection( nuclear mask ,cell type)
    
    cell_type_channels_masked = multiplex_masked[:,:,3:]                                         # 5CHNs
    X = cell_type_channels_masked.reshape(-1,5)                                                  # shape :  n_features(cell types) * n_components (multiplex_crop.shape[0])
    y = mask_crop.reshape(-1)                                                             # nuclear mask   
    perc_mask = X.sum(axis=0)/y.sum()                                                     # mask size propotion to forground area 
    min_perc = 0            # default if all 0
    if perc_mask.sum() >0:
        min_perc = perc_mask [perc_mask>0].min()
    return X,y,min_perc,cell_type_channels_masked,CTypemask_crop,perc_mask,np.array(cell_type_pixel_sum)

def gray2color(gray,color):
    img=np.zeros([gray.shape[0], gray.shape[1],3] )
    for c in range(2):
        img[:,:,c] = gray * (color[c]/255)
    return img.astype(np.uint8)

def multiplexCheck(multiplex_crop, mask_crop,Min_coef=0.4, Min_Perc = 0.27):
    '''
    @ Min_coef : minimun OMP parameter for a positive cell type channel
    @ Min_Perc:  min percentage of a cell type mask to the whole nuclear mask
                 to be claim as a positive cell 

    Outputs:
    1) If unknown class cell
      @ CHNmask_ls      = []
      @ idx_r           = []
      @ phenotype       = 0
      @ CTypemask_crop  = None
      @ coef            = None       
    2) If Single Cell
      @ CHNmask_ls      = []
      @ idx_r           = class id of positive chanels  (Non zero)
      @ phenotype       = the channel have the max perc_mask*coef
      @ CTypemask_crop  = CTypemask_crop
      @ coef            = OMP Coefficients 
    3) If Split in to N Cells ( N<=3)    
      @ CHNmask_ls      = [Cell type binary mask 1, ... Cell type binary mask N]
      @ idx_r           = class id of positive chanels (Non zero)
      @ phenotype       = the channel have the max perc_mask*coef
      @ CTypemask_crop  = CTypemask_crop
      @ coef            = OMP Coefficients  
    '''
    ###  prepareOMP  ### 
    idx_r = []
    X,y,min_perc,cell_type_channels_masked,CTypemask_crop,perc_mask,cell_type_pixel_sum =  prepareOMP(
                                    multiplex_crop,  mask_crop, CTypeParas_dic)

    if X.sum() == 0:                                                                     # No positive channle, declare a False positive  
        return [],[],0,None,None

    multiplex_crop_temp = multiplex_crop

    # incase all the cell type masks are too small, enhance to image intensity gradually
    it = 0
    while min_perc < Min_Perc and it< 20:                                               # enhance the multiplex_crop_temp
        multiplex_crop_temp = multiplex_crop_temp*1.1    # increase the intensity contrast        
        X,__,min_perc,cell_type_channels_masked,CTypemask_crop,perc_mask,cell_type_pixel_sum =   prepareOMP(
                                        multiplex_crop_temp, mask_crop, CTypeParas_dic)
        it +=1

    ###  OMP  to solve a sparsity problem : nuclear mask = sum (cell type masks)
    reg =  linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=3).fit(X,y)             # at most split into 3 cells
    coef = np.array(reg.coef_)                                                            # the raw coef of OMP
    phenotype = np.argmax(perc_mask*coef) +1                                              # phenotye result is the idx of the largest coef 
    phenotype = np.argmax(cell_type_pixel_sum) +1 if phenotype ==0 else phenotype         # in case not founded, use the maximun intensisy

    debris_num_ls = []                                                                    # the number of debrits           
    [debris_num_ls.append(measure.label (cell_type_channels_masked[:,:,ci]).max() ) for ci in range(cell_type_channels_masked.shape[2])]
    debris_nums = np.array(debris_num_ls)

    if sum(coef) > 0.8:                                                                 # all high, and min mask is large enough
        idx_r = np.where(  np.array(coef> Min_coef)  
                           & np.array(perc_mask> Min_Perc)  
                           & np.array(debris_nums < 3) 
                        )[0]
    else:                                                                               # all small, and min mask is large enough
        idx_r= np.where(  np.array(coef> (Min_coef*0.8)) 
                           & np.array(perc_mask> Min_Perc) 
                           & np.array(debris_nums < 3) 
                        )[0]
    
    CHNmask_ls = []
    for i in idx_r:
        CHNmask = cell_type_channels_masked[:,:,i]*i
        # CHNmask = morphology.convex_hull_image(CHNmask)                                   # convert to convex mask
        CHNmask_ls.append(CHNmask) 

    if  len(idx_r)>1 :                                                                      # more than one positive channels
        i_mask, u_mask =  iandu_mask(CHNmask_ls,mask_crop)
        iou= i_mask/u_mask
        if i_mask < 0.7:                                                                  # intersection of masks of positive channels 
            return CHNmask_ls,idx_r,phenotype,CTypemask_crop,coef

    return [],idx_r,phenotype,CTypemask_crop,coef

def plot_crop(label_id, Min_coef = 0.4,blobRs = [13,24], split_check = True,mask_coords_abs =None, verbose = True):
    '''
    @ Min_coef :para for multiplexCheck
    lower threshold for a positive cell type mask, the lower -> the more splitted cells, default 0.25
    @mask_coords_abs = None, (get from label image)
    @mask_coords_abs = [[x1,x2...],[y1,y2...]] , get from rle 

    Outputs:
    1) If Single Cell
      @ phenotype          = cell type class id
      @ splited_label      = None
      @ updated_phenotypes = None

    2) If Split in to N Cells ( N<=3)    
      @ phenotype          = major cell type class id
      @ splited_label      = local split segmentation label (same size as crop image, max n)
      @ updated_phenotypes = [ phenotype, phenotype2.. phenotype N ]
    '''
    split_indicator = 0     # 0 : one cell, 1 more than one cell
    splited_label = None
    updated_phenotypes = None
    Min_Perc = 0.25

    ''' Load the cropped image and masks '''
    if mask_coords_abs == None:
        min_row ,min_col,max_row,max_col = objs_dict["xmin"][label_id],objs_dict["ymin"][label_id], \
                                        objs_dict["xmax"][label_id],objs_dict["ymax"][label_id]
        multiplex_crop = multiplex[min_row:max_row, min_col:max_col,:].copy()               # raw multiplex image, 8CHN 
        gray_crop = multiplex_crop[:,:,2]
        mask_crop =  wholelabel[min_row:max_row, min_col:max_col].copy()                    # get the nuclear mask bbox
        mask_crop [mask_crop!=label_id] = 0                                                 # only get the mask 
        mask_crop = (mask_crop>0).astype(np.int)                                            # convert to binary , the full mask
    else:                                                                                   # temp, for merge use
        min_row ,min_col = mask_coords_abs[0].min(), mask_coords_abs[1].min()
        max_row ,max_col = mask_coords_abs[0].max(), mask_coords_abs[1].max()
        multiplex_crop = multiplex[min_row:max_row, min_col:max_col,:].copy()               # raw multiplex image, 8CHN 
        gray_crop = multiplex_crop[:,:,2]        
        mask_crop = np.zeros_like (gray_crop, dtype = np.int)
        mask_crop [mask_coords_abs[0] -min_row ,mask_coords_abs[1] - min_col]  = 1

    '''Criteria 1: multiplexCheck, split cells from different cell type'''
    CHNmask_ls,idx_r,phenotype,CTypemask_crop,coef = multiplexCheck(multiplex_crop, mask_crop, Min_coef,Min_Perc)
    if len(idx_r)>1:                                                                    # multiplex split
        updated_phenotypes = list( idx_r +1)                                            # new phoentypes results of splited cells individually            

    '''Criteria 2: LoGCheck, split cells from same cell type'''
    # if CHNmask_ls == [] :
    LoG_Para = [blobRs[0],blobRs[1],5,0.01,0.7]  # blobRadius_min,max,num_sigma,blob_thres,overlap
    blobs = seg_fcts.LoG_seed_detection(IMG = gray_crop, blob_LoG_Para = LoG_Para)
    numOfSameType = len( blobs )
    if  numOfSameType> 1 and len(idx_r)<1:                          # generate fg masks
        CHNmask_ls = []
        for i,(x,y) in enumerate( zip( np.uint(blobs[:,0]), np.uint(blobs[:,1]) ) ):         #blobs read from seed detection result (blobs_log) or seed table
            seed_mask = np. zeros_like(mask_crop)
            seed_mask[x,y] = 1                    # disks of seeds are label as their id (1,2,3....)
            CHNmask_ls.append(seed_mask)
        # import pdb; pdb.set_trace()
        updated_phenotypes = list([phenotype])*numOfSameType

    ''' Split '''
    if CHNmask_ls != []:
        # import pdb; pdb.set_trace()
        splited_label = watershed_newcenters( CHNmask_ls ,mask_crop,
                                    crop_img = gray_crop)            
        
        # waney mask: remove the overlap pixels from adjacent objects                       # avoid splitting by adjacent objects
        border_crop = whole_borders[min_row:max_row, min_col:max_col].copy()
        border_labels = measure.label(border_crop) * mask_crop                              # the intersection of label by borders and full mask
        waney_mask_crop = (border_labels== np.median(border_labels))              

        if  check_splited_label(splited_label*waney_mask_crop, min_perc= Min_Perc):
            split_indicator = 1                
            phenotype = updated_phenotypes[0]                                               # assign the single phenotype as the first label

            # if verbose:
            #     io.imsave(os.path.join(write_objSplit_path, str(label_id)+".png"),skimage.img_as_ubyte(splited_label) )
        else:   # split mask is in valid 
            splited_label = None
            updated_phenotypes = None
    else:
        splited_label = None
        updated_phenotypes = None

    ''' save the split cell fig for visulization '''
    if verbose==True : 
        enlarge_w = max(0, int( ( 50- multiplex_crop.shape[0])/2))                                     # enlarge the zoom-in window to show more pixles 
        enlarge_h = max(0, int( ( 50- multiplex_crop.shape[1])/2))                                    

        multiplex_view = multiplex[min_row - enlarge_w: max_row + enlarge_w,                # apply local thresholding
                                min_col - enlarge_h: max_col + enlarge_h,:].copy()
                
        if  split_indicator ==1 : #or 
            #  (split_indicator==0 and len(coef.nonzero()[0])==1 and label_id%5000==0) ):  # save split or no split every 5000
        # if plot==True :
            coef_8CHN = [0]*3+list(coef)                                                #  add 3 zeros for nuclear channels
            # fig,axs = plt.figure(dpi=120,figsize=(8,5))
           
            coef_8CHN = [0]*3+list(coef)                                                #  add 3 zeros for nuclear channels
            cols = 7
            rows = 2        
            fig,axs= plt.subplots(rows,cols, dpi=300,figsize=(8,2))
            # fig.tight_layout()
            multiplex_view = dt_utils.image_adjust(multiplex_view)
            

            mask_view = np.zeros_like(multiplex_view[:,:,0])
            mask_view[enlarge_w:enlarge_w + mask_crop.shape[0],
                      enlarge_h:enlarge_h + mask_crop.shape[1] ] = mask_crop  # mrcnn mask

            label_view = np.zeros_like(mask_view)
            label_view[enlarge_w:enlarge_w + splited_label.shape[0],
                       enlarge_h:enlarge_h + splited_label.shape[1] ] = splited_label
            original_border_view = segmentation.find_boundaries( (label_view>0)*1)
            split_border_view = segmentation.find_boundaries(label_view)

            for col_i , chn_i in  enumerate(  np.arange(2,8,1) ) :              # start from DPH channel
                
                CHN_name = CTypemultplex[chn_i]
                CHN_coef = coef_8CHN[chn_i]

                chn_view = multiplex_view[:,:,chn_i]                
                chnmask_view = np.zeros_like(multiplex_view[:,:,0])
                chnmask_view[enlarge_w:enlarge_w + mask_crop.shape[0],
                             enlarge_h:enlarge_h + mask_crop.shape[1] ] = CTypemask_crop[:,:,chn_i]  # ctype mask

                # Row 1: raw img
                if CHN_name == "DPH":
                    DPH_vis = multiplex_view[:,:,:3]
                    DPH_vis[:,:,2] = 0 
                    DPH_vis[original_border_view>0] = 255
                    axs[0,col_i].imshow(DPH_vis)    
                    
                    axs[0,col_i].set_title(CTypemultplex[chn_i] )   # multiplex n

                else:
                    # import pdb; pdb.set_trace()vis
                    color =  CTypeParas_dic[CHN_name]["c"]
                    chn_border = segmentation.find_boundaries(chnmask_view)
                    chn_border_vis= gray2color(chn_border*255,color)
                    chn_view_vis = skimage.color.gray2rgb(chn_view)
                    chn_view_vis [chn_border>0]=0
                    chn_view_vis = chn_view_vis + chn_border_vis
                    axs[0,col_i].imshow(chn_view_vis )                     
                    axs[0,col_i].set_title(CTypemultplex[chn_i] +"\n" + '{0:.1f}'.format(CHN_coef))   # multiplex n
                
                axs[0,col_i].axis("off")

                axs[1,col_i].remove()
            axs[1,6].remove()
            axs[0,6].set_title( "-".join([list(CTypeNames_dic.keys())[i] for i in idx_r])) 
            axs[0,6].axis("off")
            # axs[0,6].imshow( border_labels,cmap="gray")   #                  # splited border
            DPH_vis = multiplex_view[:,:,:3]
            DPH_vis[:,:,2] = 0 
            DPH_vis[split_border_view>0] = 255
            axs[0,6].imshow(DPH_vis)    

            plt.tight_layout(pad=0,w_pad=-5,h_pad=0.4)
            plt.savefig(os.path.join(write_objSplit_path, str(label_id)+"--vis.tif"))
            plt.close()


            
    return phenotype, splited_label,updated_phenotypes


'''Run cell spliting''' 
label_id_ls = list(objs_dict.index )
Min_coeff = float(args.thres)

# get parameters
x_len = (objs_dict["xmax"] - objs_dict["xmin"] ) 
y_len = (objs_dict["ymax"] - objs_dict["ymin"] ) 
r_avg = int( min ( x_len.mean(),y_len.mean() )  /2 ) 
r_max= min ( x_len.max(),y_len.max() )  /2

blobRs = [r_avg, (r_max+r_avg ) /2] # minr, maxr
print ("args.mp=",dt_utils.str2bool(args.mp),"Min_coeff=",Min_coeff, "[Rmax,Rmin]=",blobRs) 

if dt_utils.str2bool(args.mp):
    
    try:
        cpus = multiprocessing.cpu_count()
    except NotImplementedError:
        cpus = 2  # arbitrary default
    with multiprocessing.Pool(processes=cpus) as pool:
        r_ls = pool.starmap(plot_crop,zip(label_id_ls, repeat(Min_coeff),repeat(blobRs)))
else:    
    r_ls = [plot_crop(i, Min_coeff,blobRs) for i in label_id_ls]
phenotype_ls            = [r[0] for r in r_ls ]
splited_label_ls        = [r[1] for r in r_ls ]
updated_phenotypes_ls   = [r[2] for r in r_ls ]

t3 = time.time()
print("Used time=", int(t3-t2))
Orignial_NumOfCells= len(objs_dict.index)

print ("Original objs",Orignial_NumOfCells )
#%%
print(''' \n4. Update the final segmentation and phenotyping result''')
objs_dict["type_split"]=updated_phenotypes_ls
objs_dict["phenotype"] =phenotype_ls
objs_dict.to_csv(args.fTable_LOC)                                       # save the cell splitting indicator to orginial ftable

perf_dict = {"FP":0,"homo_split":0, "heter_split" :0}

new_dict["phenotype"] = phenotype_ls
new_dict = new_dict.set_index("ID")
new_ID = max(label_id_ls)+1                                                             # the adding cells
for i, label_id in enumerate(label_id_ls):
    splited_label        = splited_label_ls[i]                                                # the second output variable of the mp function
    updated_phenotypes   = updated_phenotypes_ls [i]                                          # the third output variable of the mp function
    p_origin = phenotype_ls[i]

    if p_origin == 0 :                                                       # update the split label

        new_dict.loc[label_id] = objs_dict.loc[label_id][[
                "centroid_x","centroid_y","xmin","ymin","xmax","ymax"]].tolist()+[0]
        perf_dict["FP"] += 1

    elif splited_label is not None:
        split_dict = measure.regionprops_table (splited_label,
                    properties=["bbox","centroid","label"] )
        x_shift ,y_shift = objs_dict["xmin"][label_id],  objs_dict["ymin"][label_id]
        
        for si , slabel in enumerate( split_dict["label"]):                              # only need to update the added one : i.e. 2,3
            # print ("updated_phenotypes=",updated_phenotypes)
            p = updated_phenotypes[si]
            xmin       = x_shift + split_dict["bbox-0"][si]
            ymin       = y_shift + split_dict["bbox-1"][si]
            xmax       = x_shift + split_dict["bbox-2"][si]
            ymax       = y_shift + split_dict["bbox-3"][si]
            centroid_x = x_shift + split_dict["centroid-0"][si]
            centroid_y = y_shift + split_dict["centroid-1"][si]
            if si == 0:                                                                 # use original label ID, change centroid,bbox
                new_dict.loc[label_id] = [centroid_x,centroid_y,xmin,ymin,xmax,ymax,p]
            else:
                new_dict.loc[new_ID] = [centroid_x,centroid_y,xmin,ymin,xmax,ymax,p]     # use new ID
                mask_coords_rel   = np.where(splited_label == slabel) 
                mask_coords_abs   = (mask_coords_rel[0] + x_shift, 
                                     mask_coords_rel[1] + y_shift)                       # absolute mask
                wholelabel [mask_coords_abs[0],mask_coords_abs[1]] = new_ID              # update the whole label
                new_ID += 1
        if np.max(updated_phenotypes) == np.min(updated_phenotypes) :
            perf_dict["homo_split" ] +=1
        else:
            perf_dict["heter_split" ] +=1

            
new_dict .to_csv(os.path.join(write_path,"bbox_phenotype.csv"))        
print("Unsupervised evaluations: ",perf_dict)


hf = h5py.File(os.path.join(write_path,"merged_labelmask.h5"), 'w')
hf.create_dataset('seg_results', data=wholelabel)
hf.close()
splited_merged_border = segmentation.find_boundaries(wholelabel)
tiff.imsave(os.path.join( write_path , "merged_border.tif"),skimage.img_as_ubyte(splited_merged_border*255))

if args.vis:
    CTypeNames_dic["None"] = 2              # phenotype = 0
    for  c_name in  CTypeNames_dic:
        seeds_chn = np.zeros_like (wholelabel,dtype=np.bool)
        CHN_id = CTypeNames_dic[c_name]
        seed_x = new_dict["centroid_x"][new_dict["phenotype"]== CHN_id -2]
        seed_y = new_dict["centroid_y"][new_dict["phenotype"]== CHN_id -2]
        diskR = 3                                   
        for i,(x,y) in enumerate( zip( seed_x, seed_y ) ):                                  #blobs read from seed detection result (blobs_log) or seed table
            seeds_chn[x,y] = 1                                                              # disks of seeds are 1
        seeds_marker = morphology.binary_dilation (seeds_chn,morphology.disk(diskR))        # sure forground (marked) is from blobs with same radius
        tiff.imsave(os.path.join(write_path,c_name+"_seeds.tif"),skimage.img_as_ubyte(seeds_marker*255))
t4 = time.time()

print("Used time=", int(t4-t3))
Updated_NumOfCells = len(new_dict.index)
print ("Updated objs", Updated_NumOfCells)

if remove_lc == True:
    for label_id in new_dict.index[ new_dict["phenotype"] == 0]:
        x_shift ,y_shift  = new_dict["xmin"][label_id],  new_dict["ymin"][label_id]
        x_max ,y_max      = new_dict["xmax"][label_id],  new_dict["ymax"][label_id]
        label_crop        = wholelabel[x_shift:x_max,y_shift: y_max]
        # mask_coords_rel   = np.where(label_crop == label_id) 
        label_crop [label_crop == label_id] = 0                      # absolute mask
        wholelabel[x_shift:x_max,y_shift: y_max] = label_crop                 # set it to 0, remove the object

    hf = h5py.File(os.path.join(write_path,"merged_labelmask_lcRemoved.h5"), 'w')
    hf.create_dataset('seg_results', data=wholelabel)
    hf.close()
    splited_merged_border = segmentation.find_boundaries(wholelabel)
    tiff.imsave(os.path.join( write_path , "merged_border_lcRemoved.tif"),skimage.img_as_ubyte(splited_merged_border*255))

t5 = time.time()

print("Remove object Used time=", int(t5-t4))
Updated_NumOfCells = len(new_dict.index)
print ("Updated objs", Updated_NumOfCells)

# perf_dict.tocsv("Split_Performance.csv")
print ("Total used time(s)=", time.time()- t0)
