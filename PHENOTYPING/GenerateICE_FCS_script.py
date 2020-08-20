# -*- coding: utf-8 -*-
"""
Input: 

    [INPUT_DIR]  : Input Images directory
    [maskType] :  input mask type options   
            Opt1 detection bouding box:  "b"/"bbox"
            Opt2 segmentation mask: "m"/"mask"
    [MaskDir]: mask directory
            Opt1 bbox: *.txt
                header contains: ['ID', 'centroid_x', 'centroid_y', 'xmin', 'ymin', 'xmax', 'ymax']
            Opt2 mask: .out /.txt
                segmentation lable mask array:  same size with image, delimited=','  
    [CHNDEF]: dataset definition file, contain dataset definition, channel definition and intensity image path ralative to input_dir  
            (Notice! All intensity images must be shown as the XML defined)             

            Opt1: None
                Look for  *datasetName*_Dataset_fromXLSX.xml  in the folder of INPUT_DIR
            Opt2:  *datasetName*.xml
                the full path of user defined dataset definition file
            Opt3: *datasetName*.csv
                the full path of user defined dataset definition file

    [saveVis] 1/0 whether or not save the 8 bit image
            save the downscaled images, recommend 1 for .ICE and .FCS, 0 for .FCS only
    [downscaleRate] int >=1
            'display_downscaleRate > 1 downscale, default = 1'
            for FCSexpress like visulization software, we normally need to  downscale (e.g.4) to reduce the RAM cost
         or else the software will crash
    [seedSize]'size of seed to create bin mask
            e.g when downscale Rate =4, seedSize = 2 for clear protion
    [erosion_px] 'integer of pixel to shrink the bbox '
            how much to shrink the bbox to get the smaller location of cells
    [wholeCellDir] optional: run whole cell body segmentation on top of nuclear segmentation
            

Output: 
  [output_dir]: 
    *datasetName*_.ice                     : ICE file with 8bit image as background
    *datasetName*__FeatureTableOnly.ice    : ICE file only with feature table
    *datasetName*__FeatureTable.bin        : ICE file supplement, since it could only recognize bin file
    *datasetName*__FeatureTable.csv        : feature table csv, only for temp file for ice
    *bimoarker*_8bit(_downscale_).tif      : downscaled imgs for all channels

e.g.
    [From bbox]  
    cd /brazos/roysam/xli63/exps/SegmentationPipeline/DataAnalysis

    data_dir="/brazos/roysam/50_plex/Set#1_S1"
    output_dir="$data_dir"/ICE_FCS_files_cell_type_images_erosion5
    mkdir "$output_dir"

    python GenerateICE_FCS_script.py \
        --INPUT_DIR="$data_dir"/cell_type_images \
        --OUTPUT_DIR="$output_dir" \
        --maskType=b \
        --MaskDir="$data_dir"/detection_results/bbxs_detection.txt \
        --downscaleRate=4 \
        --seedSize=2 \
        --erosion_px=5 \
        2>&1 | tee "$output_dir"/run-ICE-generate_bbox_erosion.txt


    python GenerateICE_FCS_script.py \
        --INPUT_DIR="/project/roysam/datasets/TBI/G2_Sham_Trained/G2_BR#22_HC_13L/final" \
        --OUTPUT_DIR=/project/roysam/datasets/TBI/G2_Sham_Trained/G2_BR#22_HC_13L/ICE \
        --maskType="b" \
        --MaskDir=/project/roysam/datasets/TBI/G2_Sham_Trained/G2_BR#22_HC_13L/detection_results/bbxs_detection.txt \
        --CHNDEF="/project/roysam/datasets/TBI/20_plex.csv" \
        --downscaleRate=4 \
        --seedSize=2 \
        --erosion_px=5 \

    [From mask]  
    python GenerateICE_FCS_script.py \
        --INPUT_DIR=/data/xiaoyang/CHN50 \
        --maskType=mask \
        --MaskDir=$data_dir \
        --OUTPUT_DIR=/data/xiaoyang/CHN50/mrcnn_seg_data/ICE_FCS_files \
        --downscaleRate=4 \
        --seedSize=3\
        2>&1 | tee /data/xiaoyang/CHN50/mrcnn_seg_data/log_GenerateICE_FCS_script.txt       
        
@author: Rebecca ( xiaoyang.rebecca.li@gmail.com ) University of Houston, 2018
"""
# pip install fcswrite
import fcswrite
import itertools
import sys
import os
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import tostring, Element, SubElement, Comment
from xml.dom import minidom
import skimage
import struct
import pandas as pd
import numpy as np
from skimage import io,transform,morphology,exposure,segmentation
from PIL import Image as Img
import warnings
import h5py
from skimage.external import tifffile as tiff

warnings.filterwarnings("ignore")

import argparse,time
# Parse command line arguments
parser = argparse.ArgumentParser(
    description='****  Whole brain segentation pipeline: FCS / ICE file generator ******' +
    '\ne.g\n' ,
    formatter_class=argparse.RawTextHelpFormatter)
    
parser.add_argument('--INPUT_DIR', required=False,
                    metavar = "/path/to/dataset/",
                    default = "/data/xiaoyang/10_plex_stroke_rat/",
                    help='Directory of inputs, default = "/data/xiaoyang/10_plex_stroke_rat/"')                
parser.add_argument('--maskType', required= True,                        
                    default = "b",  
                    help='Mask type: "b"/"bbox","m"/"mask", default = "bbox" ')                
parser.add_argument('--MaskDir', required= True,
                    default = '/data/xiaoyang/10_plex_stroke_rat/detection_results/bbxs_detection.txt',
                    help='Full Path name of mask.out/bbox.txt, default = "/data/xiaoyang/10_plex_stroke_rat/detection_results/bbxs_detection.txt" ')
parser.add_argument('--OUTPUT_DIR', required= False,
                    metavar = "/path/to/maskfile/",
                    default = '/data/xiaoyang/10_plex_stroke_rat/',
                    help='Root directory for the results, default = "/data/xiaoyang/10_plex_stroke_rat/" ')                        
parser.add_argument('--CHNDEF', required= False,
                    metavar = "/path/to/csvFeatureFile/",
                    default = None,
                    help='localtion for xmlfile or csv file, if not provide then default to localize the INPUT_DIR')          
parser.add_argument('--ftable', required= True,
                    default = None,
                    help='Full Path name of user defined featuretable, if None then automatically generate intrinsic featureby mask/bbox')


parser.add_argument('--saveVis', required= False,
                    type = int,
                    default = 1,
                    help= 'save8bit image for display or not, defalut = 0')          
parser.add_argument('--downscaleRate', required= False,
                    type = int,
                    default = 1,
                    help= 'display_downscaleRate > 1 downscale, < 1 upscale, default = 1')  
parser.add_argument('--seedSize', required= False,
                    type = int,
                    default = 1,
                    help= 'size of seed to create bin mask')                                                                               
parser.add_argument('--erosion_px', required= False,
                    type = int,
                    default = 0,
                    help= 'integer of pixel to shink the bbox ')
                                     
parser.add_argument('--imadjust', required=False,
                    default = "T",type = str, 
                    help='whether to adjust the image, ')         

# whole cell body reconstruction
parser.add_argument('--wholeCellDir', required= False,
                    metavar = "/path/to/maskfile/",
                    default = '/data/xiaoyang/10_plex_stroke_rat/',
                    help='Root directory of whole cell body segmentation')   
parser.add_argument('--debug', required= False,
                    metavar = "whether to debug",
                    default = "T",type = str, 
                    help='whether to debug ')      
args = parser.parse_args()


def image_adjust( image):
    if image.ndim ==3:
        CHN_img_ls = []
        for i in range(image.shape[2]):
            CHN_img = image [:,:,i]
            CHN_img_ls. append( exposure.equalize_adapthist(CHN_img) )           # Load over images
        image = np.dstack (CHN_img_ls)
    else:
        image = exposure.equalize_adapthist(image)            # Load over images
    image = skimage.img_as_ubyte(image)
    return image.copy()

def extractFileNamesforType(dir_loc, fileExt):   # dir , '.tif'
    readNames =    os.listdir (dir_loc)
    types = np.array( [ ([ x.find(fileExt)>0 for x in  readNames ]) ] , dtype = int)               # only extract keywords
    typesIDs = np.array  (  np.where(types > 0)[1] )
    fileNames = []
    ([fileNames.append(readNames[i]) for i in typesIDs])    
    return fileNames

def save_mask2bin (maskBinName, label_img, transpose = True):
    tiff.imsave(maskBinName.split(".bin")[0] + ".tif", label_img )                              # save the tiff version of bin file
    
    label_img_vec = np.array(label_img, dtype = np.int32)
    if transpose:
        label_img_vec = np.reshape( label_img_vec.T ,label_img_vec.size)               
    else:
        label_img_vec = np.reshape( label_img_vec ,label_img_vec.size)                  
    fout_mask = open(maskBinName,'wb')    
    label_img_vec.tofile(fout_mask)                       
    fout_mask.close()

def str2bool(str_input):
    bool_result = False if str_input.lower() in ["f",'false','0',"no",'n'] else True
    return bool_result

def Write_FeatureTable (Read_img_file_Loc, maskfileName, Write_img_file_Loc = None, MaskType = 'bbox' ,ftable=None,
                        display_downscaleRate = 1, seedSize = 1, saveVis = False, mirror_y = True,erosion_px=0,
                        wholeCellDir = None, CHNDEF = '' ,imadjust = False):
    ''' 
    Function: Write_FeatureTable
    Input: 
        [Read_img_file_Loc] 
            location of intensity images and DatesetDef [**XLSX.XML] 
            The xml file contain dataset definition, channel definition and intensity image path  
        Notice! All intensity images must shown as the XML defined (.tif),each (N, M) ndarray cd ..       
        [MaskType]
            'bbox','mask'     
        [maskfileName = Bbox_txt_name.txt] full path name of the bbox txt
            ['ID', 'centroid_x', 'centroid_y', 'xmin', 'ymin', 'xmax', 'ymax']        
        [maskfileName = MaskLabel.out] full path name of labeled image 
            same size with image, delimited=','    
        [ftable] user defined feature table, if None, automatically generated intrinsic feature by mask
            ftable must have the columns name "biomarker"

        [display_downscaleRate]  int
            only open to "mask" option        
                display_downscaleRate > 1 downscale
                display_downscaleRate < 1 upscale
        [saveVis] bool
            True for save 8bit for display
            False for not saving
        [erosion_px]: (optional) integer of pixel to shink the mask/bbox   (recommend 0 to 5)

    Output:
        [**_FeatureTable.txt]
    ''' 

    ''' Inputs parameter loading''' 
    print ( "----Write_FeatureTable----")
    if Write_img_file_Loc == None:
        Write_img_file_Loc = Read_img_file_Loc
    else:
        if os.path.exists(Write_img_file_Loc) is False :
            os.mkdir (Write_img_file_Loc)
    print ("saveVis = ",  saveVis )
    print ("MaskType = ",  MaskType )
    print ("maskfileName = ",  maskfileName )

    print ("    ''' Load Dataset definition'''")
    if ".xml" in CHNDEF:
        tree = ET.parse(CHNDEF)
        root = tree.getroot()
        Images = root[0]        
        Image_temp = Images.findall('Image')[0]
        imreadName = Image_temp.find('FileName').text  # Read one image to extract the size of image
    elif ".csv" in CHNDEF:
        # Create xml structure from csv, for loading the channel definition later
        DatesetDef_csv = pd.read_csv(CHNDEF)
        imreadName = DatesetDef_csv["filename"][0]    # Read one image to extract the size of image
        Images = Element('Images')        
        for i in range(DatesetDef_csv.shape[0]):
            Image = SubElement(Images, 'Image')
            biomarkerName = DatesetDef_csv["biomarker"][i]                   
            Image.set('biomarker',biomarkerName) 
            FileName = SubElement(Image, 'FileName')
            FileName.set("CHNName", DatesetDef_csv["biomarker"][i])
            FileName.text = DatesetDef_csv["filename"][i]

    datasetName = os.path.basename(CHNDEF).split('.')[0]  # storage the name to extract dataset name 
    with tiff.TiffFile(os.path.join( Read_img_file_Loc, imreadName)) as tif:
        image_temp = tif.asarray(memmap=True)
    print ("    ''' read xml done'''")
    print ("FileName= ",os.path.join( Read_img_file_Loc, imreadName))
    print ("    ''' read image_temp done'''")

    image_shape = image_temp.shape
    
    print ("    ''' downscale'''")    
    image_downscaled_temp = transform.pyramid_reduce(image_temp, downscale=display_downscaleRate) if display_downscaleRate > 1 else image_temp
    downscales_shape = image_downscaled_temp.shape        
    print ( "image.shape  = ", image_shape)
    print ( "downscales_shape.shape  = ", downscales_shape)

    ''' OptionaL'''
    if ftable is not None:
        ftable_csv = pd.read_csv(ftable)

    '''Load bbox (.txt) or mask ('.out') to write centroids '''

    if  ('b' in MaskType):        # Use bbox as input  contain ['ID', 'centroid_x', 'centroid_y' ,'xmin', 'ymin', 'xmax', 'ymax''] into bbox      
        if ('.txt' in maskfileName):                            # txt file
            Bbox_headers = open(maskfileName).readline().split("\n")[0].split("\t")
            Bbox_txt     = np.loadtxt(maskfileName, skiprows=1)
            NumOfObj     = Bbox_txt.shape[0]
            assert len(Bbox_headers)== Bbox_txt.shape[1]    
            bbox_table      = {}
            for i, Bbox_header in enumerate( Bbox_headers):    
                # Storage ['ID', 'centroid_x', 'centroid_y' ,'xmin', 'ymin', 'xmax', 'ymax''] into bbox                                                                   
                bbox_table[Bbox_header] = np.array(Bbox_txt[:,i],dtype= int)

        elif ('.csv' in maskfileName):                          # csv file
            bbox_table = pd.read_csv(maskfileName)            
            NumOfObj     = bbox_table.shape[0]

        # Create feature table for ICE  (will have y mirror for visualization purpose)
        Featuretable = pd.DataFrame({ 'ID'        : bbox_table['ID'],
                                    'centroid_x': bbox_table['centroid_x'],
                                    'centroid_y': bbox_table['centroid_y'],
                                    })
        Featuretable = Featuretable.set_index('ID')

        # Create Associative feature table for further analaysis, e.g. cell try training  
        AssociativeFtable = pd.DataFrame({   'ID'        : bbox_table['ID'],
                                            'centroid_x': bbox_table['centroid_x'],
                                            'centroid_y': bbox_table['centroid_y'],
                                            'xmin'      : bbox_table['xmin'],
                                            'ymin'      : bbox_table['ymin'],
                                            'xmax'      : bbox_table['xmax'],
                                            'ymax'      : bbox_table['ymax'],
                                    })
        AssociativeFtable = AssociativeFtable.set_index('ID')  
        Featuretable['centroid_y'] = Featuretable['centroid_y'] if mirror_y == False \
                                   else ( image_shape[0] - Featuretable['centroid_y'])                  # Might need to  flip over the crop 
        
        if display_downscaleRate >0 : # we need to downscale for visualize in FCSexpress with downscale , only down scale the centroid seeds
             
            ########## generate  label_image_downscaled ############
            label_image_downscaled = np.zeros(downscales_shape)     

            centroid_x_downscaled_ls = np.zeros (len(Featuretable))
            centroid_y_downscaled_ls = np.zeros (len(Featuretable))
            for i, obj_id in  enumerate(Featuretable.index)  :           
                x = int ( Featuretable['centroid_x'][obj_id] / display_downscaleRate )
                y = int ( Featuretable['centroid_y'][obj_id] / display_downscaleRate )    
                centroid_x_downscaled_ls[i] = x
                centroid_y_downscaled_ls[i] = y if mirror_y == False \
                                   else ( int ( image_shape[0] /display_downscaleRate ) - y)            # Might need to  flip over the crop 
                # make sure is [y,x] rather than [x,y]
                label_image_downscaled[y,x]  = obj_id                                                   # disks of seeds are label as their id (1,2,3....)                
            
            if seedSize > 1: 
                diskR = seedSize
                label_image_downscaled = morphology.dilation (label_image_downscaled,morphology.square(diskR))
                                                                                                        # sure forground (marked) is from blobs with same radius
            
            if display_downscaleRate > 1 :      # downscale for visualize in FCSexpress                 # with downscale , only down scale the centroid seeds
                # record seed coords
                Featuretable['centroid_x_ds'] = centroid_x_downscaled_ls                                   
                Featuretable['centroid_y_ds'] = centroid_y_downscaled_ls if mirror_y == False \
                    else ( int ( image_shape[0] /display_downscaleRate )- centroid_y_downscaled_ls )    # Might need to  flip over the crop 

            ############ save the mask_bin_downscales ################
            label_image_downscaled = label_image_downscaled if mirror_y == False else np.flipud(label_image_downscaled)
            maskBinName = os.path.join (Write_img_file_Loc , '[DAPI]Seeds_Mask.bin' )    

            hf = h5py.File(os.path.join(Write_img_file_Loc,"label_image_downscaled.h5"), 'w')
            hf.create_dataset('seg_results', data=label_image_downscaled)
            hf.close()

            save_mask2bin (maskBinName , label_image_downscaled, transpose = False)   
            
    elif ('m' in MaskType): # Use mask as input
        if display_downscaleRate == 1 :            # save whole cell reconstruction             
            print ("wholeCellDir =============", wholeCellDir)
            if wholeCellDir is not None:    # make sure the label is the same the the seeds
                for fName in os.listdir (wholeCellDir) :
                    if "outline.txt" in fName:                                                          # e.g soma_outline.txt in the same directroy with 
                        print ("loading fName ....",fName)
                        key = fName.split("_outline.txt")[0]
                        label_image_wholecell = np.loadtxt( os.path.join(wholeCellDir, fName), delimiter = ',' ,dtype = int )            
                        maskBinName = os.path.join (Write_img_file_Loc , key +  '_Mask.bin' )
                        save_mask2bin (maskBinName , label_image_wholecell)   

        if ('.out' in maskfileName) or ('.txt' in maskfileName):  # load nuclear mask
            label_image = np.loadtxt( maskfileName, delimiter = ',' ,dtype = int )

        elif  ('.h5' in maskfileName) : #load whole cell image 
            hf = h5py.File(maskfileName,'r')
            label_image = np.array(hf.get('mask')).astype(np.uint)              
            hf.close()

        '''  Generate AssociateFtable'''
            
        obj_props = skimage.measure.regionprops (label_image)
        NumOfObj = len(obj_props)        
        print ("NumOfObj=", NumOfObj, "range :",label_image.max())
        ID_ls = np.zeros (NumOfObj).astype("int")      
        centroid_x_ls = np.zeros (NumOfObj)    
        centroid_y_ls = np.zeros (NumOfObj)    
        for i, obj in enumerate( obj_props ) :
            ID_ls[i]= obj.label 
            centroid_x_ls[i] = int( obj.centroid[0]  )
            centroid_y_ls[i] = int( obj.centroid[1]  )  

        ICE_id_ls, __, __ = segmentation.relabel_sequential(ID_ls)             # reorder the masks_ids

        Featuretable = pd.DataFrame({ 'ID' : ID_ls,                         # absolute ID
                                      'centroid_x': centroid_x_ls,
                                      'centroid_y': centroid_y_ls,
                                      'ICE_id' : ICE_id_ls
                                    })
        Featuretable = Featuretable.set_index('ID')

        # Associative feature table for further analaysis, e.g. cell type training     
        AssociativeFtable = pd.DataFrame({  'ID'        : ID_ls,
                                            'centroid_x': centroid_x_ls,
                                            'centroid_y': centroid_y_ls,
                                            'ICE_id' : ICE_id_ls
                                    })
        AssociativeFtable = AssociativeFtable.set_index('ID')

        ##########################################################
        centroid_x_downscaled_ls = np.zeros (NumOfObj,dtype = int)    
        centroid_y_downscaled_ls = np.zeros (NumOfObj,dtype = int)       

        # Create downscaled label_image only with centroids
        label_image_downscaled = np.zeros(downscales_shape)    
        for i, obj in enumerate( obj_props ) :
            # ID_ls[i]= obj.label                 
            x = int ( obj.centroid[0] / display_downscaleRate )
            y = int ( obj.centroid[1] / display_downscaleRate ) if mirror_y == False \
                else ( int ( image_shape[1] /display_downscaleRate )- int ( obj.centroid[1] / display_downscaleRate )  ) # Might need to  flip over the crop 
            centroid_x_downscaled_ls[i] = x
            centroid_y_downscaled_ls[i] = y                                 
            # label_image_downscaled[ x, y] = obj.label                                  # disks of seeds are label as their id (1,2,3....)
            label_image_downscaled[ x, y] = ICE_id_ls[i]                                  # disks of seeds are label as their id (1,2,3....)

        # generate  label_image_downscaled
        if seedSize > 1: 
            diskR = seedSize
            label_image_downscaled = morphology.dilation (label_image_downscaled,morphology.disk(diskR))    # sure forground (marked) is from blobs with same radius
        # save the mask_bin_downscales
        maskBinName = os.path.join (Write_img_file_Loc , 'Seeds_Mask.bin' )              # [DAPI]Seeds_Mask.bin
        save_mask2bin (maskBinName , label_image_downscaled, transpose = False)           
        print ("downscaled NumOfObj=", len( np.unique(label_image_downscaled))-1, "range :",label_image_downscaled.max())
        # import pdb; pdb.set_trace()

        # record seed coords
        if display_downscaleRate != 1 :      
            Featuretable['centroid_x_ds'] = centroid_x_downscaled_ls 
            Featuretable['centroid_y_ds'] = centroid_y_downscaled_ls 
        
        #  Featuretable['centroid_y_ds'] = centroid_y_downscaled_ls if mirror_y == False \
                                    # else ( int ( image_shape[0] /display_downscaleRate )- centroid_y_downscaled_ls )   # Might need to  flip over the crop        
        
        # for mask, we need to mirror x rather than y
        Featuretable['centroid_x'] = centroid_x_ls        
        Featuretable['centroid_y'] = centroid_y_ls if mirror_y == False \
                                        else ( image_shape[1] - centroid_y_ls )                                # Might need to  flip over the crop 
        print ("Featuretable['centroid_y'] range:" ,   Featuretable['centroid_y'].min(), Featuretable['centroid_y'].max() )
    else:
        raise ValueError("[Error!] Fails to load bbox /mask files")

    '''Load images one by one to generate image display tif and Featuretable for each channel'''
    for Image in  Images.findall('Image'):    
        biomarker = Image.get('biomarker')     
        print ("biomarker = ", biomarker)                                                           # Gain biomarker from XML
        if biomarker is 'NONE' :       
            print ("Your DatesetDef_XML need biomarker")
        else:                                                                                   
            ChannelName = '[' +biomarker+']'                
            print ("Loading Channel : ", biomarker )            
            imreadName = Image.find('FileName').text
            print ("imreadName = ",os.path.join( Read_img_file_Loc, imreadName))
            with tiff.TiffFile(os.path.join( Read_img_file_Loc, imreadName)) as tif:
                image = tif.asarray(memmap=False)            

            saveImageName = "[" + Image.get('biomarker')  + "].tif" 
            ''' Low quality Image for visualization  '''
            if saveVis == True:     
                if imadjust:                
                    image_save =  image_adjust(image)          # Load over images
                                                                                # save 8 bits image for display
                if display_downscaleRate == 1 : 
                    image_save  = skimage.img_as_ubyte(image)
                    tiff.imsave (os.path.join( Write_img_file_Loc , saveImageName),
                                    image_save )               
                else:
                    image_downscaled = transform.pyramid_reduce(image, downscale=display_downscaleRate)
                    # print ("image_downscaled.shape =",image_downscaled.shape )         
                    image_save  = skimage.img_as_ubyte(image_downscaled)
                    tiff.imsave (os.path.join( Write_img_file_Loc , saveImageName),
                                    image_save)    
                    print (" Save downscaled Images! " )

            ''' Storage Intrinsic features '''
            # import pdb;pdb.set_trace()

            if ftable is not None and (biomarker) in ftable_csv.keys():      
                Featuretable[(biomarker) ]  = ftable_csv[(biomarker) ]                
                print ("Channel",(biomarker)," used user defined features")
            else:
                Avg_intensity_ls = []
                if MaskType == 'bbox':
                    print ("Computing biomarker Avg and Sum")
                    
                    obj_i_ls = [ i for i in range(0,NumOfObj) ]
                    def cal_avg_intensity(i):
                        # print (bbox_table["xmin"][i],bbox_table["xmax"][i],bbox_table["ymin"][i],bbox_table["ymax"][i] )

                        y_min = int( np.array( [bbox_table["ymin"][i] + erosion_px,bbox_table["ymax"][i]- erosion_px]). min() )
                        y_max = int( np.array( [bbox_table["ymin"][i] + erosion_px,bbox_table["ymax"][i]- erosion_px]). max() )
                        x_min = int( np.array( [bbox_table["xmin"][i] + erosion_px,bbox_table["xmax"][i]- erosion_px]). min() )
                        x_max = int( np.array( [bbox_table["xmin"][i] + erosion_px,bbox_table["xmax"][i]- erosion_px]). max() )
                        intensity_image_cropped  = image[ y_min : y_max  ,
                                                        x_min: x_max  ] 
                        # Tol_intensity_ls[i] = intensity_image_cropped.sum()
                        return np.mean (intensity_image_cropped)
                    Avg_intensity_ls = [cal_avg_intensity(i) for i in obj_i_ls]
                    Avg_intensity_ls = np.array(Avg_intensity_ls)

                elif MaskType == 'mask':
                    print ("Computing biomarker Avg and Sum")
                    intensity_image_ls = skimage.measure.regionprops_table (label_image,intensity_image = image,
                                        properties=["intensity_image"] )["intensity_image"]
                    Avg_intensity_ls = [np.mean(i) for i in intensity_image_ls]
                    Avg_intensity_ls = np.array(Avg_intensity_ls)        

                Featuretable[(biomarker) ] = Avg_intensity_ls
                AssociativeFtable[(biomarker) ] = Avg_intensity_ls
                # Featuretable[(biomarker +'__Avg') ] = Avg_intensity_ls
                # Featuretable[(biomarker +'__Sum') ] = Tol_intensity_ls

    '''Write Associative Feature table and Feature Table into csv....... for futher analysis /ICE '''
      
    AssociativeFtable_FName = os.path.join( Write_img_file_Loc, datasetName + '_AssociativeFtable.csv' )
    AssociativeFtable.to_csv(AssociativeFtable_FName)

    featureTable_FName = os.path.join( Write_img_file_Loc, datasetName + '_FeatureTable.csv' )
    Featuretable.to_csv(featureTable_FName)

    return featureTable_FName, AssociativeFtable_FName 

def Write_FeatureTable2bin(featureTable_FName):        
    '''write it into                            .bin ........................ '''
    featureVariableMat = pd.read_csv(featureTable_FName)
    #     export the real feature Values data, generate into bin file     
    featureVariable_float = np.array(featureVariableMat, dtype = np.float32)
    featureVariableLs_float = np.reshape( featureVariable_float.T ,featureVariable_float.size)
    
    # write into .bin file
    tableBinName = featureTable_FName.split('.csv')[0] + '.bin'
    fout = open(tableBinName,'wb')               
    featureVariableLs_float.tofile(fout)                       
    fout.close()
    tableBinSize = os.path.getsize(tableBinName)        
    ##print (tableBinSize)
    if tableBinSize != (len(featureVariableLs_float) *32 /8) :                                  # the size of the binary file shall equal to ( #of objects * bitdepth) /8
        print ('[Error !] FeatureTable into binary image wrong')
    else:
        print ('featureTable bin file has written in', tableBinName)


def GenerateICE(datasetName, output_folder, featureTable_FName = None,FeatureTableOnly = False):
    '''
    Function : GenerateICE
    Input:
    [input_folder]
        % [externel ..def.xml ]file to get the channel information #
    [output_folder] should already contain    
        % [**_featuretable.csv] Feature table from NucleusEditor 
        % [**_featuretable.bin] Feature table from NucleusEditor 
        % [original image]  illuminaionCorrectedImage (temp  % use'4110_C1_IlluminationCorrected_stitched.tif'   )
        % [Seeds_Mask.bin]
    Output:
        % [.ice] XMLfile
    '''        

    def prettify(elem):
        """Return a pretty-printed XML string for the Element.
        """
        rough_string = tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")
    
    ###################################  Read input files ###########################
    ''' Load Dataset definition'''
    actualImageNames = []
    maskImageNames = []

    for fileName in sorted( os.listdir(output_folder) ):
        if os.path.isfile(os.path.join(output_folder,fileName)) :                                                        # not folder
            if ('[' in fileName) and ('.tif'in fileName)  :
                actualImageNames.append(fileName)
            if ('Mask' in fileName) and ('.bin'in fileName)  : #mask_....bin
                maskImageNames.append(fileName)
    print("\nmaskImageNames=",maskImageNames)

    # read feture table
    # Read the external feature table.txt
    if featureTable_FName is None: 
        featureTable_FName = extractFileNamesforType(output_folder, '_FeatureTable.csv')[0]     
        fo = open( os.path.join(output_folder ,  featureTable_FName) )                           #Read featuretable from .txt generated from Nucleus Editor
    else:
        fo = open(  featureTable_FName )                                                         #Read featuretable from .txt generated from Nucleus Editor

    content  = fo.readlines()                                                                    # the full data of the table 
    fo.close()
    content_Mat =  ([x.split(',') for x  in content])   
 
    featureVariableMat = content_Mat[1:len(content_Mat)]                                         # row[1:end]: extract the feature names
    
    featureNameLs = content_Mat[0]                                                               # row[0]:     FeatureNames 
    if featureNameLs[len(featureNameLs)-1] == '\n' :                                             # if the last element of featureNameLs is "\n", delete the last one
        featureNameLs.remove('\n')  
        ([x.remove('\n')    for  x  in featureVariableMat])        
    
    #for defXMLName in defXMLNames :                                      FN            
                
    #####################################  Generateing ICE  ##########################
    rootET = Element('ICEFormat')
    rootET.set('xmlns','http://www.isac-net.org/std/ICEFormat/1.0/ice')
    rootET.set('xmlns:xsi','http://www.w3.org/2001/XMLSchema-instance')
    rootET.set('xsi:schemaLocation','http://www.isac-net.org/std/ICEFormat/1.0/ \n http://flowcyt.sf.net/ice/ICE.xsd')
    rootET.set('version','1.0')
    comment = Comment('Description of the contents of the ICEFormat Data Directory file')
    rootET.append(comment)
    
    #### Parsing Channel Definitions    
   
    ### FeatureDefinitions share among all datasets
    FeatureDefinitions = SubElement(rootET, 'FeatureDefinitions')        
    
    PrimitiveFIDs = []
    for FID, featureName in enumerate(featureNameLs):                                           #featureNameList read from .txt 
        if len(featureName)>1:
            FeatureDefinition = SubElement(FeatureDefinitions, 'FeatureDefinition')       
            InfoFloat = SubElement(FeatureDefinition, 'InfoFloat')
            Description = SubElement(InfoFloat, 'Description')
            Description.text = featureName                                                      # call for featureName
            ID = SubElement(InfoFloat, 'ID')
            ID.text = 'F0'+str(FID)
            PrimitiveFIDs.append(ID.text)
            BitDepth = SubElement(InfoFloat, 'BitDepth')
            BitDepth.text = str(32)     
    
    
     ###   Dataset    different animals are in different dataset
    Dataset = SubElement(rootET, 'Dataset')                
    Dataset.set('Info', '10 plex stroke rat')

    ##   featureTable                                                                
    Metadata = SubElement(Dataset, 'Metadata')
    NumberOfObjects = SubElement(Metadata,'NumberOfObjects')                       
    NumberOfObjects.text = str(len(featureVariableMat))                                         # Number of Objects = Number of Rows ub featureVariableLs
    Custom = SubElement(Metadata,'Custom')
    DataGeneratedBy = SubElement(Custom,'DataGeneratedBy')
    Make = SubElement(DataGeneratedBy,'Make')
    Make.text = 'MRCNN TransferLearning Instance Segmentation'
    
    
    ## FeatureValues       # define the featurevalue save file
    FeatureValues = SubElement(Dataset, 'FeatureValues')
    
    FeatureValue  = SubElement(FeatureValues, 'FeatureValue')                                   # feature Value for actual features
    Primitive =  SubElement(FeatureValue, 'Primitive')
    URL = SubElement(Primitive,'URL')
    featureTableBinname = featureTable_FName.split('.csv')[0] + ('.bin')
    URL.text = ('file://'+ os.path.basename(featureTableBinname))                                                 # get the file Name of Image        
    for PrimitiveFID in PrimitiveFIDs:                                                          # read ele 
            FeatureID = SubElement(Primitive, 'FeatureID')
            FeatureID.text = PrimitiveFID               
    
    if FeatureTableOnly == False:
        ## images
        CompositeImages = SubElement(Dataset, 'CompositeImages')
        CompositeImages.append(Comment('Association with channels definition'))
        
        # ImageIDs_others = []
        Biomakers_others = []

        # only read the size once , because all channels have same sized
        with tiff.TiffFile(os.path.join( output_folder , actualImageNames[0])) as tif:
            actualImg = tif.asarray(memmap=True)
        # actualImg = tiff.imread( os.path.join( output_folder , actualImageNames[0]))            # pic one for extracting the meta
        actualImg_shape = [actualImg.shape[1],actualImg.shape[0]]

        
        for ImgID , actualImageName in enumerate(actualImageNames):
            Image = SubElement(CompositeImages,'Image')
            ID = SubElement(Image,'ID')
            ID.text = 'Img_00' + str(ImgID)                                                     # only keep the name part before extention as the ID 
            URL = SubElement(Image,'URL')
            URL.set('url','file://'+ actualImageName)                                           # get the file Name of Image    
            Width = SubElement(Image,'Width')
            Width.text = str( actualImg_shape[0])                                              
            Height = SubElement(Image,'Height')
            Height.text = str(actualImg_shape[1])                                                   
            
            biomarkerName = actualImageName.split('_')[0]                                       # prepare for InfoCompositeImage
            Image.set('biomarker',biomarkerName) 
            # if biomarkerName == '[DAPI]':                                              
            #     ImageID_DAPI = ID.text
            # else:
            #     ImageIDs_others.append( ID.text )                                               # for use later in InfoCompositeImage

        ##   Masks    [read the mask image .bin external file]
        Masks = SubElement(Dataset, 'Masks')
        Masks.append(Comment('Association with segmentation and feature value definition'))    
        
        MaskIDs_seeds =[]
        MaskIDs_others =[]
        for MaskID , maskImgName in enumerate(maskImageNames):
            #individual Mask images
            Mask = SubElement(Masks,'Mask')
            ID = SubElement(Mask,'ID')
            ID.text = 'Mask_00' + str(MaskID)                                                   # mask ID correspond to image and channel
        #    MaskID.text = ID.text                                                              # correspond to ImageID in InfoCompositeImage
            URL = SubElement(Mask,'URL')
            URL.text = ('file://'+  maskImgName)                                                # get the file Name of MaskImage (Should be .bin file)
              # Read Image size
            Width = SubElement(Mask,'Width')
            Width.text = str( actualImg_shape[0])
            Height = SubElement(Mask,'Height')
            Height.text = str(actualImg_shape[1])
            BitDepth = SubElement(Mask, 'BitDepth')
            BitDepth.text = str(32)                                                             # unsigned integer 32 bit
            
            # biomarkerName = maskImgName.split(']')[0] +']'                                    # prepare for InfoCompositeImage
        #    Mask.set('biomarker',actualImageName.split('_')[0]) 
            if "Seeds" in maskImgName :                                                         # nuclues Mask, generate for all channels
                MaskIDs_seeds.append(ID.text)
                # print ("MaskIDs_seeds:", MaskIDs_seeds)
            else:                                                                               # morphological Mask  # might us in the future        generate for all other channels
                MaskIDs_others.append(ID.text)
                # print ("MaskIDs_others:", MaskIDs_others)
                             
        # exit dataset-----------------------------
  
        CompositeImageFIDs = []        # save the CompositeImageFIDs for FeatureValue stack holder
        # Generate Composite Image for DAPI channel and DAPI masks (nucleus_Mask, wholeCell_Mask and cytoplasm_Mask)
        COMP_ID = 1
        for Image in  CompositeImages.findall('Image'):                                         # for each channel Img
            ImgID = Image.find('ID').text                                                       # image ID 
            ImgBiomaker = Image.get('biomarker')                                                # biomarker (channel Name)   '[]'
            
            for Mask in  Masks.findall('Mask'):                                                 # for each mask   (find previous definition)
                MaskID = Mask.find('ID').text      
                
                if MaskID in MaskIDs_seeds:                                                   # generate(nucleus_Mask, wholeCell_Mask and cytoplasm_Mask) for all channels
                    MaskID_seeds = MaskID
                    
                    FeatureDefinition = SubElement(FeatureDefinitions, 'FeatureDefinition')       
                    InfoCompositeImage = SubElement(FeatureDefinition, 'InfoCompositeImage')     # will define  later after we read Image and masks
                    Description = SubElement(InfoCompositeImage, 'Description')
                    # MaskDescription_seeds = Mask.find('URL').text.split('.bin')[0].split("]")[1]   # get mask type  from file name ,e.g."nuclues_Mask"
                    # Description.text = ImgBiomaker + ' with ' + MaskDescription_seeds        # [Channel] in mask
                    Description.text = ImgBiomaker # [Channel] in mask
                    
                    ID = SubElement(InfoCompositeImage, 'ID')
                    ID.text = 'F0_COMP_'+ str(COMP_ID)                                           #create the compond image ID
                    COMP_ID = COMP_ID + 1
                
                    CompositeImageFIDs.append(ID.text)            
                    ImageID = SubElement(InfoCompositeImage, 'ImageID')                          # will define later after we read Image
                    ImageID.text = ImgID
                    
                    MaskID = SubElement(InfoCompositeImage, 'MaskID')                            #cal_avg_intensity            # will define later after we read mask
                    MaskID.text = MaskID_seeds                                                 # biomarker (channel Name)   '[]'
                
                elif MaskID in MaskIDs_others:                                                   # assciate its morphological Mask
                    MaskBiomaker =  Mask.find('URL').text.split('[')[1] + '['                    # get channel name from file name ,e.g."[S100]"
                    MaskBiomaker = MaskBiomaker.split(']')[0] + ']'
                    if MaskBiomaker == ImgBiomaker:                                              # only associate the morphology mask with the corresponding channel
                        MaskID_other =  MaskID
                        FeatureDefinition = SubElement(FeatureDefinition, 'FeatureDefinition')       
                        InfoCompositeImage = SubElement(FeatureDefinition, 'InfoCompositeImage') # will define  later after we read Image and masks
                        Description = SubElement(InfoCompositeImage, 'Description')
                        
                        MaskDescription_seeds = Mask.find('URL').text.split(']')[1] 
                        MaskDescription_seeds = MaskDescription_seeds.split('.bin')[0]       # get mask type  from file name ,e.g."morpholgoical_Mask"
                        Description.text = ImgBiomaker + ' with ' + MaskDescription_seeds      # [Channel] in mask
                        
                        ID = SubElement(InfoCompositeImage, 'ID')
                        ID.text = 'F0_COMP_'+ str(COMP_ID)                                       #create the compond image ID
                        COMP_ID = COMP_ID + 1
                    
                        CompositeImageFIDs.append(ID.text)            
                        ImageID = SubElement(InfoCompositeImage, 'ImageID')                      # will define later after we read Image
                        ImageID.text = ImgID
                        
                        MaskID = SubElement(InfoCompositeImage, 'MaskID')                        # will define later after we read mask
                        MaskID.text = MaskID_other      

        # supplemetry for feature Values of Composite Image (because the ID have to be define previousely)
        ### feature Value for CompositeImage
        FeatureValue = SubElement(FeatureValues, 'FeatureValue')
        CompositeImage =  SubElement(FeatureValue, 'CompositeImage')                            # for stack holder
        for CompositeImageFID in CompositeImageFIDs:                                            # read ele 
                FeatureID = SubElement(CompositeImage, 'FeatureID')
                FeatureID.text = CompositeImageFID       
      
        ### write into xml file
        xml_fileName = datasetName + '.ice'
    else:                                                                                       # FeatureTableOnly = True       
        CompositeImageFIDs = []                                                                 # save the CompositeImageFIDs for FeatureValue stack holder
        COMP_ID = 1            
        FeatureValue = SubElement(FeatureValues, 'FeatureValue')       
        xml_fileName = datasetName + '_FeatureTableOnly.ice'

    xml_file = open( os. path.join ( output_folder , xml_fileName),'w')    
    xml_file.write(prettify(rootET))
    xml_file.close()

    return  os. path.join ( output_folder , xml_fileName)

def GenerateFCS(csv_filename, fcs_filename = None):
    ''' Function of CSV to FCS credit to Jahandar '''
    df = pd.read_csv(csv_filename)
    df = df.dropna(axis=1)
    

    ids = df[['ID']]
    centers = df[['centroid_x', 'centroid_y']]
    phenotypes = df.drop(['ID', 'centroid_x', 'centroid_y'], axis = 1)                          # Drop columns

    data = np.hstack((ids.values.astype(int), centers.values.astype(int), phenotypes.values))
    channel_names = [item for item in [list(ids), list(centers), list(phenotypes)]]
    channel_names = list(itertools.chain.from_iterable(channel_names))

    if fcs_filename is None:
        fcs_filename = csv_filename.split(".csv")[0] + ".fcs"

    fcswrite.write_fcs(filename=fcs_filename,
                       chn_names=channel_names,
                       data=data)


if __name__== "__main__":
 
    tic = time.time ()
    assert args.maskType in ["b", "bbox", "m", "mask"]
    mask_type = 'bbox' if  args.maskType in ["b", "bbox"] else 'mask'
    print ("** mask_type = ", mask_type)

    if args.CHNDEF is None:
        DatesetDef_path = extractFileNamesforType(args.INPUT_DIR, 'XLSX.xml')[0]
    else:
        DatesetDef_path =  args.CHNDEF

    print ("** args.MaskDir = ", args.MaskDir)
    featureTable_FName, AssociativeFtable_FName = Write_FeatureTable ( Read_img_file_Loc       = args.INPUT_DIR , 
                                                    maskfileName            = args.MaskDir  ,
                                                    Write_img_file_Loc      = args.OUTPUT_DIR,
                                                    ftable                  = args.ftable,
                                                    MaskType                = mask_type,
                                                    display_downscaleRate   = args.downscaleRate,
                                                    seedSize                = args.seedSize,
                                                    saveVis                 = bool(args.saveVis),                                                     
                                                    mirror_y                = True,
                                                    wholeCellDir            = args.wholeCellDir,
                                                    CHNDEF                  = DatesetDef_path,
                                                    imadjust                = str2bool(args.imadjust)
                                                   )
    print ("Finish creating featureTable_FName = ", featureTable_FName,
                    "\t AssociativeFtable_FName= ", AssociativeFtable_FName)

    Write_FeatureTable2bin(featureTable_FName)
    
    # only featuretable, no mask superimposing
    # ice_filename1 = GenerateICE(input_folder = args.INPUT_DIR, output_folder = args.OUTPUT_DIR,  
    #                             featureTable_FName = featureTable_FName ,  FeatureTableOnly = True)

    datasetName = os.path.basename(DatesetDef_path).split('.')[0]
    ice_filename2 = GenerateICE(datasetName = datasetName, output_folder = args.OUTPUT_DIR,  
                                featureTable_FName = featureTable_FName ,  FeatureTableOnly = False)   
    print ("ICE has written in ", ice_filename2)

    # GenerateFCS(csv_filename = featureTable_FName)

    toc = time.time ()
    print ("Total Time(h) : ", (toc-tic)/3600)