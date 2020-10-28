'''
Murge the detection result
Inputs:
    [dataset_dir]:
        raw crop images ,
        size : 300 * 300 px,
        name : xmin_ymin
        [submit.csv]
            storage the rle information of all the segmentation result in this folder
    [img]:
        whole brain Img  
Outputs:
    [Seg_vis.tif]
    [whole_image.tif]
    [submit_all.csv]
Author, Rebecca (xiaoyang .rebecca.li@gmail.com)  University of Houston ECE dept, Farsight Lab 2018
e.g.
    # load grouped results
    python merge_detection_result.py -g 1 \
        -i /uhpc/roysam/xiaoyang/exps/NuclearSeg_DNN/Mask_RCNN/datasets/CHN50_div/DPH_molded_div/ \
        -s merge_test_temp/submit \
        -o merge_test_temp \
        -re 0
    # load submit in one folder
    python merge_detection_result.py -g 0 \
        -i /uhpc/roysam/xiaoyang/exps/NuclearSeg_DNN/Mask_RCNN/datasets/CHN50_mold/DPH_molded/ \
        -s merge_test_temp/submit \
        -o merge_test_temp \
        -re 1 \
    # Merge result only, no need to stitch images
    python merge_detection_result.py -g 0 \
        -i /uhpc/roysam/xiaoyang/exps/NuclearSeg_DNN/Mask_RCNN/datasets/CHN50_mold/[DAPI+HISTONES]16bit.tif \
        -s merge_test_temp/submit \
        -o merge_test_temp_new \
        -re 1 \
            
'''
import sys,os,cv2
import numpy as np
import shutil
import skimage
from  skimage import segmentation,morphology,filters 
import skimage.io as io
import csv
import datasets_utils as dt_utils
import pandas as pd
from skimage.external import tifffile as tiff
import h5py

class Paras(object):
    def __init__(self):
        # Parameters for ORB
        # self.Group          = False
        self.RemoveEdge     = True  
        self.MergeMethod    = "mask"
        self.crop_size      = [512,512]
        self.crop_overlap   = 50
        self.img_shape      = [29286,42906]                                                                 # only use when merge crop images
  
    # def set_Group(self, Group):
    #     self.Group    = bool(Group)
    def set_RemoveEdge(self, RemoveEdge):
        self.RemoveEdge    = bool(RemoveEdge)
    def set_img_shape(self, img_shape):
        self.img_shape    = img_shape
    def set_crop_size(self, crop_size):
        self.crop_size    = crop_size

    def display(self):
        print ("====================\n")
        print( "Parameters  : ")
        # print ("\Group = "          , self.Group)
        print ("\RemoveEdge = "     , self.RemoveEdge)
        print ("\MergeMethod = "    , self.MergeMethod)
        print ("\crop_size = "      , self.crop_size)
        print ("\img_shape = "      , self.img_shape)
        print ("\crop_overlap = "   , self.crop_overlap)

def check_coord_shape(coord,shape):
    # make sure all the x and y coordinate are all within the range of the shape
    coord_checked = [[],[]]
    check_bool_x = coord[0] < shape[0] 
    check_bool_y = coord[1] < shape[1] 
    check_bool_xy = check_bool_x*check_bool_y

    coord_checked = [ coord[0][check_bool_xy], coord[1][check_bool_xy]]
    return coord_checked

def result_merge(whole_image, submit_file,  write_folderName=None, paras=None,local=False,verbose=0):
    # def result_merge(whole_image, submit_files_dir,  write_folderName, paras):
    cropimg_fullPath_ls = []
    if verbose:
        print ("loading submit.csv:",submit_file )
        print (''' load submit , create rle_dic ''')
    rle_dic         = {}                                                                                 #  {"img_id", "[],[],...""}
    score_dic       = {}
    class_dic       = {}
    obj2imgID_dic      = {}                                                                                 #  {"img_id", "[2,3,4]""}
    obj2imgID_ls_merged       = []

    assert ".csv" in submit_file
    csvfile         = open(submit_file, 'r')
    if verbose:
        print ("loading submit.csv:",submit_file )
    spamreader      = csv.reader(csvfile, delimiter=',')
    key_visited_ls  = []        
    for rowId, row in enumerate( spamreader ) :                                                     # load img_id,
        if rowId > 0 :                                                                              # skip first line
            if len(row)>=2 and row[1]!= '':                                                         # only read the img_id have more than one object detected
                img_id = row[0]            
                # initialize
                if img_id not in  key_visited_ls:                                                   # initialize the dic  (notice dic could should have exclusive keys)
                    key_visited_ls.append (img_id) 
                    rle_dic[img_id] = []                                                         
                    score_dic[img_id] = []  
                    class_dic[img_id] = []  
                # add value
                if len (row[1]) > 2 :  # at least one object been detected
                    rle_dic[img_id] .append ([row[1]] )                                                # load all rles for the images
                    if len(row) > 2:                                                                   # If score and class has been written in submit
                        score_dic[img_id].append ( np.float( row[2] ) )
                        class_dic[img_id].append ( np.uint8( row[3] ) )


    csvfile.close()      
    if verbose:
        print ("*****Finished load ",str(rowId), "Rows",len(rle_dic),)
        print ("*****result_merge********")

    if local :
        paras       = Paras()
        paras.RemoveEdge= True
        paras.crop_overlap  = 0
        paras.crop_size  = whole_image.shape[:2]        
    # elif "FeatureTable.csv" in submit_file:   # not recommend ,when rles is not written
    #     print ("loading FeatureTable.csv:",os.path.join(submit_files_dir,submit_file) )
    #     featureTable_df = pd.read_csv(os.path.join(submit_files_dir,submit_file))
    #     key_visited_ls  = []
    #     for obj_id in range(featureTable_df.shape[0]):
    #         img_id = featureTable_df["cropimg_ID"][obj_id]
    #         if featureTable_df["area"][obj_id]>0:                                                       # only load the valid detected objects
    #             if img_id not in  key_visited_ls:
    #                 key_visited_ls.append (img_id)                                                      # initialize the dic  (notice dic could should have exclusive keys)
    #                 obj2imgID_dic[img_id] = []                                                          
    #             obj2imgID_dic[img_id] .append (obj_id )                                                 # load all rles for the images
    #     print ("*****Finished load Rows",len(obj2imgID_dic))

     # load whole brain image  and merge segmentation result without coping with cropped images

    if type (whole_image) == dict:      # load from folder, converted to dic
        wholeImage = whole_image[0].copy()

    else :#type ( whole_image) == np.ndarray or mmep
        wholeImage = whole_image
        
    img_shape       = (wholeImage.shape[0], wholeImage.shape[1])
    border_color    = [0,255,0] if wholeImage.ndim == 2 else [255,255,255]                                  # border color: green if image is gray ; white if image is colored 
    
    merged_visual   = wholeImage[:,:,:3].copy() if wholeImage.ndim == 3 else skimage.color.gray2rgb(wholeImage).copy()
    merged_label    = np.zeros((img_shape[0], img_shape[1]),dtype=int)
    merged_borders  = np.zeros_like(merged_label,dtype=np.bool)
    merged_submission = []    
    crop_width, crop_height    = paras.crop_size
    img_rows, img_cols         = img_shape                                                                  # img_rows = height , img_cols = width    
    # Min_intensity = merged_visual.mean()*0.3                                                             # default:0.5  # the obj mean smaller would be delete
    # Max_iou = 0.4                                                                                       #default:0.3

    obj_id = 0
    objids_ls = []
    scores =[]
    classes = []
    objids_ls.append(0)
    scores.append(0)                                                                                    # label 0 is for background
    classes.append(0)
    rle_out = False

    if rle_out==True:
        rle_ls = []

    for img_id in rle_dic.keys():
        if local is True:  # not shifting the location of image 
            i=j=0
        else:
            i = int(img_id.split("_")[0])                                                                       # min_row,min_max of bbox
            j = int(img_id.split("_")[1])  
            # import pdb; pdb.set_trace()

        # define a window to check the labels in neighbor cropped images.with size of [2*crop_overlap +crop_height,2*crop_overlap +crop_width]
        suspicious_shift_height = paras.crop_overlap  if i> paras.crop_overlap  else 0
        suspicious_shift_width  = paras.crop_overlap  if j> paras.crop_overlap  else 0
        suspicious_label_window = merged_label[ i-suspicious_shift_height : i + crop_height + paras.crop_overlap , 
                                                j-suspicious_shift_width  : j + crop_width  + paras.crop_overlap ]         # extract the labels of subpicious window from previous merged results
        # import pdb; pdb.set_trace()

        if len ( rle_dic[img_id] )> 0  :    # image level                                                               
            for rle_id,rle in enumerate( rle_dic[img_id] ):   # object level 
                # rle                         = rle_dic[img_id]
                rle = rle[0]
                if type(rle) == list:
                    rle = rle[0]
                crop_mask                   = dt_utils.rle_decode(rle, paras.crop_size)                         # relative mask ,same size as crop_size       
                if paras.RemoveEdge is True:
                    crop_mask               = dt_utils. cleanEdgedmask( crop_mask,2)                          # remove mask if it is at border
                
                if crop_mask.sum()>100 :# and convex_perc > 0.7 :                                                 # pre-check the size of mask, need to be big enough
                    mask_coords_rel   = np.where(crop_mask)                                                       # relative mask coordinated 
                    border_coords_rel = np.where(segmentation.find_boundaries(crop_mask))
                    mask_coords_abs   = (mask_coords_rel  [0] + i, mask_coords_rel  [1] + j)                      # absolute mask
                    border_coords_abs = (border_coords_rel[0] + i, border_coords_rel[1] + j)    

                    mask_coords_suspicious = (mask_coords_rel[0] + suspicious_shift_height -1, 
                                              mask_coords_rel[1] + suspicious_shift_width  -1)                               

                    # make sure the coords are inside the window (ep.for the window on the edge)
                    mask_coords_suspicious = check_coord_shape(coord = mask_coords_suspicious ,shape = suspicious_label_window.shape)
                    mask_coords_abs        = check_coord_shape(coord = mask_coords_abs        ,shape = merged_label.shape)
                    border_coords_abs      = check_coord_shape(coord = border_coords_abs      ,shape = merged_label.shape)

                    if len(mask_coords_suspicious) > 0 :# and intensity_mean > 0:      # avoid objects too dim 
                        # check_result = dt_utils.nms_check(suspicious_label_window, 
                        #                                    mask_coords_suspicious, max_iou = Max_iou) 

                        # check_result = True if (0 in suspicious_label_window.shape) else nms_check_result
                        visited_pixel = merged_label  [mask_coords_abs[0],mask_coords_abs[1]]
                        visited_pixel = (visited_pixel>0).sum()                          # pixels has been labeled in the same location from previous result

                        if visited_pixel/crop_mask.sum() < 0.3:                          # pass the nms check , add to merge label
                            obj_id = obj_id + 1    
                            objids_ls.append(obj_id)
                            merged_label  [mask_coords_abs[0],mask_coords_abs[1]] =  obj_id                           # assign pixel of the obj to the obj_id                   
                            # import pdb; pdb.set_trace()
                            merged_borders[border_coords_abs[0],border_coords_abs[1]] =  True  
#                             print (obj_id)
                            # add mereged score and class     
                            if score_dic != {} and class_dic!={}:
                                if len (score_dic[img_id] ) > 0 and len (class_dic[img_id] ) > 0:
                                    obj_score = score_dic[img_id][rle_id]
                                    obj_class = class_dic[img_id][rle_id]
                                    scores.append(obj_score)
                                    classes.append(obj_class) 
                            # save the local RLE to the merged submission                             
                                merged_submission.append("{}, {},{},{}".format(img_id, rle,obj_score, obj_class))
                            else:
                                merged_submission.append("{}, {},{},{}".format(img_id, rle))

    merged_label_max = merged_label.max()
    if verbose:
        print("merged_label.max()=", merged_label.max()) 

    merged_label,__,inv = segmentation.relabel_sequential(merged_label)     # relabel

    if write_folderName is not None:
        if verbose:
            print ("......save  merged results......")
        if os.path.exists(write_folderName) is False:
            os.mkdir(write_folderName)        
        # save label_mask
        hf = h5py.File(os.path.join(write_folderName,'merged_labelmask.h5'), 'w')
        hf.create_dataset('seg_results', data=merged_label)
        hf.close()
        # np.savetxt (os.path.join(write_folderName, "merged_labelmask.txt"),merged_label , fmt ='%d',delimiter=',',)
        merged_submission = "ImageId,EncodedPixels,Score,Class\n" + "\n".join(merged_submission)
        file_path = os.path.join(write_folderName, "merged_submission.csv")
        with open(file_path, "w") as f:
            f.write(merged_submission)        
        # save border
        # border_abs = morphology.binary_dilation(border_abs, morphology.disk(1))                    # make border thicker
        border_merged = merged_borders.astype(np.uint8)    # gray2rgb
        border_merged = border_merged*255
        tiff.imsave(os.path.join(write_folderName, "merged_result_borders.tif"), border_merged )   
        if verbose:
            print ("border_merged.max()=",border_merged.max())

        # # Save visualF
        # border_abs = np.where (merged_borders) 
        # merged_visual [border_abs[0],border_abs[1],:] =   border_color
        # tiff.imsave(os.path.join(write_folderName, "merged_result_Visual.tif"), merged_visual)

        # save feature table

        if type (whole_image) == dict:
            wholeImage = whole_image[0]
            wholeImage = np.dstack([wholeImage])
            featureTable_df_merged = dt_utils.generate_featureTable_allCHN ( merged_label, whole_image )
        else:# type ( whole_image) == np.ndarray or np.memmap:
            featureTable_df_merged = dt_utils.generate_featureTable_allCHN ( merged_label, wholeImage )
        if verbose:
            print( "featureTable_df_merged .shape=", featureTable_df_merged .shape) 

        # add columns for class and score:
        if len (scores) > 0 and len (classes) > 0:
            # 
            relabeled_score_ls = np.array(scores [:merged_label_max+1])[inv[1:]]
            relabeled_class_ls = np.array(classes[:merged_label_max+1])[inv[1:]]
            featureTable_df_merged["score"] = relabeled_score_ls
            featureTable_df_merged["class"] = relabeled_class_ls
        
        featureTable_df_merged.to_csv(os.path.join(write_folderName, "fTable_merged.csv"))

        # # save clump_analysis table
        # clumps_df = dt_utils.clump_detection(merged_label)
        # clumps_df.to_csv(os.path.join(write_folderName, "clumps_table.csv"))

    return merged_label


if __name__ == '__main__':
    import argparse
    import time

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='****  Whole brain segentation pipeline on DAPI + Histone Channel : Merge cropped segmentation results******' ,
        formatter_class=argparse.RawTextHelpFormatter)

    # parser.add_argument('-g', '--Group', required = False,
    #                     default = False, type = int,
                        # help= " 1 to load trained results in divided groups, 0 to load for one ")
    parser.add_argument('-i','--img_LOC', required=False,
                        metavar = "/path/to/dataset/",
                        default = "/uhpc/roysam/xiaoyang/exps/NuclearSeg_DNN/Mask_RCNN/datasets/CHN50_div/DPH_molded_div/Group_0",
                        help='Root directory of cropped images to be merged, default = "/uhpc/roysam/xiaoyang/exps/NuclearSeg_DNN/Mask_RCNN/datasets/CHN50_div/DPH_molded_div/Group_0"')
    parser.add_argument('-s','--submit_LOC', required=False,
                        metavar = "/path/to/dataset/",
                        default = "/uhpc/roysam/xiaoyang/exps/NuclearSeg_DNN/Mask_RCNN/results/nucleus/submit_20181026T173132-[test]_DRHG_8bit_Group_0/",
                        help='Root directory of detection result to be merged, default = "/uhpc/roysam/xiaoyang/exps/NuclearSeg_DNN/Mask_RCNN/results/nucleus/submit_20181026T173132-[test]_DRHG_8bit_Group_0/"')
    parser.add_argument('-o', '--output_dir', required= False,
                        metavar = "/path/to/output/",
                        default = "merge_test_temp" ,
                        help='Root directory for the results, default = os.getcwd() , "merge_test_temp"')
    parser.add_argument('-re', '--RemoveEdge', required = False,
                        default = 1, type = int,
                        help= " 1 to remove componnets at the edge of cropped images , 0 not remove ")
    parser.add_argument('-a','--imadjust', required=False,
                        default = '0',type = str, 
                        help='whether to adjust the image')                                

    args = parser.parse_args()

    tic         = time .time()

    ''' Input variables from command input
    '''    
    # load a path, read then into dic
    if os.path.isdir(args.img_LOC):
        image_dict = {}
        for i, image_path in enumerate( os.listdir(args.img_LOC) ) :
            if ".tif" in image_path:                                    
                with tiff.TiffFile(os.path.join(args.img_LOC , image_path)) as tif:
                    image_dict [i]  = tif.asarray(memmap=True,)
                # image_dict [i] = tiff.imread(os.path.join(args.img_LOC , image_path)).copy()
        whole_image = image_dict
    # load a image
    else:
        whole_image = skimage.img_as_ubyte( tiff.imread(args.img_LOC  ))
        if dt_utils.str2bool(args.imadjust):                
            whole_image =  dt_utils.image_adjust(whole_image)          # Load over images

    submit_LOC = args.submit_LOC
    output_dir = args.output_dir
        
    paras       = Paras()
    # paras.set_Group     (args.Group)
    paras.set_RemoveEdge(args.RemoveEdge)
    paras.display()
    
    # print ("args.Group = ", paras.Group,", args.RemoveEdge = ",paras.RemoveEdge)

    ''' Run merging
    '''
    
    merged_label = result_merge(   whole_image      = whole_image , 
                                    submit_file     = args.submit_LOC ,
                                    write_folderName = output_dir,
                                    paras            = paras,
                                    verbose          =1)

    toc         = time .time()
    print ("total time (s) =" ,  toc - tic)