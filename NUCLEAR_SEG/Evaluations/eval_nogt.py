#!/usr/bin/env python
# coding: utf-8# Unsupervised evaluation metrixs

# In[1]:
import matplotlib
# Agg backend runs without a display
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# %matplotlib notebook

import os, sys
import numpy as np
import skimage
import tifffile 
from skimage import segmentation,io,img_as_ubyte,measure,morphology
from skimage.segmentation import find_boundaries
from sklearn import mixture
import pandas as pd
import multiprocessing
from multiprocessing.pool import ThreadPool
from skimage.transform import resize
from skimage.external import tifffile as tiff

from scipy import ndimage as ndi
import time
import argparse,time
import matplotlib
import h5py
sys.path.insert(0, "/project/ece/roysam/xiaoyang/exps/SegmentationPipeline/mrcnn_Seg/supplement")
import datasets_utils as dt_utils


from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from sklearn.cluster import SpectralClustering
'''Coverage Caculate '''
def coverage(wholelabel,multiplex_image):
    multiplex = multiplex_image.copy()
    # caluculate the coverate of the detected foreground by rough binarization    #
    for ch_id in range (multiplex.shape[2]):
        gray_img = multiplex[:,:,ch_id]
        noise_thres = dt_utils.bgnoise_est(gray_img)
        # Remove all the pixels smaller than this threshold  (denoised_img)
        multiplex[:,:,ch_id] [ gray_img < noise_thres] = 0    

    fullmask = wholelabel>0
    # Caculate the coverage percentage over dilations
    prop_dict={}
    prop_dict["dilateR"] = []
    prop_dict["prec"] = []

    for dilateR in range(5):
        if dilateR > 0:
            fullmask = morphology.dilation (fullmask,morphology.disk(1))  
        prop = (multiplex*np.dstack([fullmask]*3)).sum() / multiplex.sum()
        prop_dict["dilateR"].append(dilateR) 
        prop_dict["prec"] .append(prop)
    return pd.DataFrame.from_dict(prop_dict)

'''TSNE_cluastering Analysis '''
def zeropadding(img, canvas_size = (67,67) ) :
    # canvas size must larger than img size
 
    if len(img.shape) ==3 :
        img = img [: canvas_size[0],: canvas_size[1],:]
        paddedImg = np.zeros( (canvas_size[0],canvas_size[1],3))
        paddedImg[  0 : img.shape[0],
                    0 : img.shape[1],:] = img
        
    else :
        img = img [: canvas_size[0],: canvas_size[1]]
        paddedImg = np.zeros((canvas_size[0],canvas_size[1]))
        paddedImg[  0 : img.shape[0],
                    0 : img.shape[1]] = img
              
    return  paddedImg       

def tsne_clustering(label_img, intensity_image,output_dir= None, sample_rate = 0.01, class_table=None):
    # prepare the dataset
    regionprops = measure.regionprops_table(label_img,
                            intensity_image =intensity_image,
                            properties=('label', 'intensity_image') )
    label_df = pd.DataFrame({ "label": regionprops["label"]})
    NumofObj = len(label_df)
    random_num =  int( NumofObj*sample_rate)

    # Generate IDs       
    # data_dict = {}
    data_ID_dict = {}
    np.random.seed(0)
    data_ID_dict["random"] = np.random.choice( regionprops['label'],size=random_num, replace=False)

    if class_table is not None:    # get the seg label id for each phenotype
        class_table = pd. read_csv(class_table)
        for class_type in  pd.unique( class_table["phenotype"])  :
            # data_dict[class_type] = [] # np.zeros( (int( NumofObj*sample_rate), 67*67))    
            data_ID_dict[class_type] = class_table.loc[class_table["phenotype"] == class_type,["ID"]].values.flatten()


    hf = h5py.File('tsne_data.h5', 'w')

    for key in data_ID_dict.keys():
        X = []
        for data_ID in list( data_ID_dict[key] ):
            id  = int(label_df.loc [ label_df["label"] == data_ID ].index.values)
            crop_img = regionprops["intensity_image"][id]    
            crop_img_zeropadded = zeropadding(crop_img, [67,67])     
            X.append( crop_img_zeropadded.reshape(-1) )

        # t-SNE embedding of the digits dataset
        X = np.stack ( X  )
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)    
        X_tsne = tsne.fit_transform(X)

        hf.create_dataset(str(key), data=X_tsne)
        if output_dir is not None:
            plt.figure()
            for i in range(X_tsne.shape[0]):
                plt.scatter(X_tsne[i, 0], X_tsne[i, 1] ,s=1,cmap="k")
            plt.grid()
            plt.savefig(os.path.join(output_dir, "TSNE_"+ str(key) +".png"), bbox_inches="tight", dpi=300)
            plt.close()
            print("Computing t-SNE embedding for",key, " sample size", len( X ))


    hf.close()

            

'''Clump_detection '''
def clump_detection(label_img, output_dir= None):
    bin_label = (label_img>0)
    bin_label= morphology.binary_erosion(bin_label,  morphology.disk(2))   # make mask thinner for weakly connected ones
    clump_label = measure.label( bin_label*1,connectivity=1 )
    clumps=[]    
    for obj_id , obj in enumerate( measure.regionprops(clump_label)):     
        clump_pixels = label_img[obj.coords[:,0],obj.coords[:,1]] 
        if len( np.unique( clump_pixels ) )> 1 :  
            bg_pixels=(clump_pixels ==0).sum()                            # the area that it never been visited
            convex_diff = obj.convex_image.sum() - obj.area               #the area that it dfference from the convec         
            if bg_pixels <100 :                               
                clump_crop,__,__ = segmentation.relabel_sequential(label_img[obj.bbox[0]:obj.bbox[2], obj.bbox[1]:obj.bbox[3]])
                clumps.append( len( np.unique(clump_pixels) ) )
    df_clumps = pd.DataFrame(data={ "clump_nums": clumps },dtype=np.int)
    # print the df_centroid_shift
    if output_dir:
        plt.figure()
        df_clumps.hist(bins=len(np.unique(clumps)))
        # plt.hist(clumps_ls, 20, density=True)
        plt.xlabel("Connected Objects")
        plt.ylabel("Density")
        plt.title("Histogram of Clump numbers " +"\n" +
                "mClump = {0:.3f}".format(np.mean(clumps)) + 
                "$\pm${0:.3f}".format(np.std(clumps)) )    
        plt.grid()
        plt.savefig(os.path.join(output_dir, "histClump_final.png"), bbox_inches="tight", dpi=300)
        plt.close()
    return df_clumps


if __name__ == '__main__':
    
    
    tic = time.time()   
    parser = argparse.ArgumentParser(
        description='Calculate IOU over 2 whole brain label mask')

    parser.add_argument('--output_dir','-o' ,required= False,
                        metavar = "/path/to/maskfile/",
                        default = 'iou_df.csv',
                        help='Full name to save as result ')                          
    parser.add_argument('--pd', required=True,
                        metavar="/path/to/dataset/",
                        default = "/project/hnguyen/xiaoyang/exps/Data/50_plex/jj_final/seg_results/autoseg/merged_labelmask.h5",
                        help='pd_labels txt')   
    parser.add_argument('--img', required=False,
                        metavar="/path/to/dataset/",    
                        default = "/project/hnguyen/xiaoyang/exps/Data/50_plex/jj_final/images_stacked/RDGHBDPH.tif",
                        help='Load image to crop objs , not required')
    parser.add_argument('--class_table', required=False,default= None,
                        metavar="/path/to/phenotype table/",    
                        help='Load phenotype table , not required')
    args, _ = parser.parse_known_args()

    # In[4]:

    if ".npy" in args.pd:
        wholelabel = np.load( args.pd )
    elif ".txt" in args.pd:            
        wholelabel = np.loadtxt( args.pd ,delimiter=",", dtype = int)
    elif ".h5" in args.pd: 
        hf = h5py.File(args.pd, 'r')                                                             # load wholelabel use 9s
        wholelabel = np.array(hf.get('seg_results'))
        hf.close()
    print(args.pd)

    with tiff.TiffFile(args.img) as tif:  
        multiplex =  tif.asarray(memmap=False)[:,:,:3]

    # import pdb;pdb.set_trace()
    # multiplex =  tifffile.imread(args.img)
    os.makedirs(args.output_dir,exist_ok=True)
    ''' Coverage'''
    # df_coverage = coverage(wholelabel,multiplex)
    # df_coverage.to_csv(os.path.join( args.output_dir, "coverage_table.csv"))

    ''' TSNE'''
    tsne_clustering(wholelabel,multiplex[:,:,2],
                output_dir= args.output_dir,
                class_table = args.class_table)
    # ''' clump '''
    # df_clump = clump_detection(wholelabel)
    # df_clump.to_csv(os.path.join( args.output_dir, "clump_table.csv"))
