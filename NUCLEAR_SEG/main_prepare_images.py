'''
raw_dir="/project/hnguyen/datasets/50_plex/S3/original"
dataset_dir="/project/hnguyen/xiaoyang/exps/Data/50_plex/S3"

python main_prepare_images.py \
--INPUT_DIR=$raw_dir \
--OUTPUT_DIR=$dataset_dir/images_stacked_multiplex \
--DAPI S3_R2C1.tif \
--HISTONES S3_R2C2.tif \
--NEUN S3_R2C4.tif \
--S100 S3_R3C5.tif \
--OLIG2 S3_R1C9.tif \
--IBA1 S3_R1C5.tif \
--RECA1 S3_R1C6.tif
    
'''

import os
import sys
import time
import argparse
from skimage.external import tifffile as tiff
import skimage
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
sys.path.insert(0, os.path.join( os.getcwd(), "mrcnn_Seg"))
from supplement import datasets_utils as dt_utils

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

def imSummation (img1,img2, outputformat = '16bit') :  # save size  
    imSum = np.add(img1, img2)
    imSum = imSum/ imSum.max()                  # float    
    if outputformat ==  '16bit':
        imSum = skimage.img_as_uint(imSum)   # change to 16bit 
    else:
        imSum = skimage.img_as_ubyte(imSum)   # change to 8bit 
    return imSum

def load_image(dir_path, filename, outputformat = "8bit"):
    with tiff.TiffFile(os.path.join(dir_path,filename)) as tif:
        wholeImage = tif.asarray(memmap=False)    
    if outputformat ==  '16bit':
        wholeImage = skimage.img_as_uint(wholeImage)   # change to 16bit 
    else:
        wholeImage = skimage.img_as_ubyte(wholeImage)   # change to 8bit 
    return wholeImage
    
def prepare_multiplex(input_dir,  file_names):
    """
    Prepare training dataset file (data.h5) to train CapsNet for cell type classification
    :param input_dir: Path to the input dir containing biomarker images
    :param bbxs_file: Path to the bbxs_detection.txt file generated from cell nuclei detectio module
    :param file_names: List of filnames for channels in the order: [dapi, histone, neun, s100, olig2, iba1, reca1]
    :return:
    """
    # prepare dict for biomarkers
    # for not-existing channel put ''

    if len(file_names) ==7:
        biomarkers = {'DAPI': file_names[0],
                    'Histones': file_names[1],
                    'NeuN': file_names[2],
                    'IBA1': file_names[3],
                    'Olig2': file_names[4],
                    'S100': file_names[5],
                    'RECA1': file_names[6]}
        CTypeNames_dic = {"DAPI":0,"Histones":1,"DPH":2,"NeuN":3, "IBA1":4,"Olig2":5,"S100":6,"RECA1":7}                          # cell type names and their channel id at multiplex
        multiplex = [""]*8   # 8 chn images
    elif len(file_names)==2:    
        biomarkers = {'DAPI': file_names[0],
                    'Histones': file_names[1]}
        CTypeNames_dic = {"DAPI":0,"Histones":1,"DPH":2}                          # cell type names and their channel id at multiplex
        multiplex = [""]*3   # 3 chn images

    for i , biomarker in enumerate( CTypeNames_dic.keys() ) :
        chn = CTypeNames_dic[biomarker]    # CHN_id in the mutltiplex
        if biomarker in biomarkers.keys():
            multiplex[chn]  = load_image(input_dir, biomarkers[biomarker] ,"8bit")
            min_shape = multiplex[chn].shape
            if i == 0 :
                Min_shape = list(min_shape)
            Min_shape[0] = min ( min_shape[0] ,  Min_shape[0])
            Min_shape[1] = min ( min_shape[1] ,  Min_shape[1])
        else:
            multiplex[chn] = np.zeros_like(multiplex[0])   #DPH

    # check shape
    for m in range(len(multiplex)):
        multiplex[m] = check_shape(multiplex[m],Min_shape)
    multiplex = np.dstack(multiplex)
    #...... adjust
    multiplex[:,:,2] = imSummation(multiplex[:,:,0],multiplex[:,:,1], "8bit" )
    multiplex_adjusted = dt_utils.image_adjust(multiplex)                        # Load over images
    RDGH_visEnhanced = multiplex_adjusted[:,:,:3].copy()
    RDGH_visEnhanced[:,:,2] =0
    return multiplex_adjusted,RDGH_visEnhanced

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--INPUT_DIR', type=str, help='/path/to/input/dir')
    parser.add_argument('--OUTPUT_DIR', type=str, help='/path/to/output/dir')
    parser.add_argument('--DAPI', type=str, default='', help='<dapi.tif> | None')
    parser.add_argument('--HISTONES', type=str, default='', help='<histones.tif> | None')
    parser.add_argument('--NEUN', type=str, default='', help='<NeuN.tif> | None')
    parser.add_argument('--S100', type=str, default='', help='<S100.tif> | None')
    parser.add_argument('--OLIG2', type=str, default='', help='<Olig2.tif> | None')
    parser.add_argument('--IBA1', type=str, default='', help='<Iba1.tif> | None')
    parser.add_argument('--RECA1', type=str, default='', help='<RECA1.tif> | None')
    parser.add_argument('-a','--imadjust', required=False,
                    default = '1',type = str, 
                    help='whether to adjust the image')                                                            
    
    args = parser.parse_known_args()[0]
    os.makedirs( args.OUTPUT_DIR,exist_ok= True)

    if args.NEUN =='' or args.IBA1=='':
        multiplex_adjusted,RDGH_visEnhanced = prepare_multiplex(args.INPUT_DIR, 
                    [args.DAPI, args.HISTONES])                
    else:
        multiplex_adjusted,RDGH_visEnhanced = prepare_multiplex(args.INPUT_DIR, 
                    [args.DAPI, args.HISTONES, args.NEUN, args.IBA1, args.OLIG2, args.S100, args.RECA1])
        
    # import pdb; pdb.set_trace()


    print (multiplex_adjusted[:,:,2].max())
    tiff.imsave (os.path.join(  args.OUTPUT_DIR ,"RDGH.tif") , RDGH_visEnhanced )
    tiff.imsave (os.path.join(  args.OUTPUT_DIR ,"DPH.tif") , multiplex_adjusted[:,:,2] )
    tiff.imsave (os.path.join(  args.OUTPUT_DIR ,"multiplex.tif") , multiplex_adjusted )

