'''
Data Preparation of Images for MRCNN 

Author: Rebecca LI, University of Houston, Farsight Lab, 2018
xiaoyang.rebecca.li@gmail.com

e.g.
    ################# nucleus seg ##################
    $ source activate Rebeccaenv
    $ project_root="/brazos/roysam/xli63/exps/SegmentationPipeline/mrcnn_Seg"
    data_root="/brazos/roysam/xli63/exps/Data/50_plex/jj_final"
    subset="whole"
    dataset_dir="$project_root"/datasets/50_plex/multiplex/"$subset"
    mkdir "$dataset_dir"

    # prepare training set
    python3 DataPrepare_mrcnn.py train \
    -i "$img_dir" \
    -l "$label_dir" \
    -r "$result_dir" \
    --multiprocess=1 \
    --save_display=1 \
    --masks_fName=masks_autosegPseed \
    --imadjust=0

    ################# celltype classification ##################
    project_root="/brazos/roysam/xli63/exps/SegmentationPipeline/mrcnn_Seg"
    data_root="/brazos/roysam/xli63/exps/Data/50_plex/jj_final"
    subset="whole"

    cd "$project_root"/supplement
    python3 DataPrepare_mrcnn.py train \
    --img="$data_root"/images_stacked_multiplex/multiplex.tif \
    --label="$data_root"/images_stacked_multiplex/merged_labelmask.txt \
    --class_df="$data_root"/multiclass/roughPhenotyping/"$subset"/CellTypeTable_final.csv \
    --results="$dataset_dir" \
    --save_display=0 \
    --imadjust=0
'''

import sys,os,cv2, time
import numpy as np
import skimage
import skimage.measure, skimage.io,skimage.color
from skimage.measure import regionprops
from skimage import exposure
# import tifffile as tiff
# from PIL import Image
import warnings
warnings.filterwarnings("ignore")
# import threading
import multiprocessing
from multiprocessing.pool import ThreadPool
# import merge_detection_result as merge_crop_fcts
import datasets_utils as dt_utils
import tifffile 
import pandas as pd

from display_training import *  # display_instances...
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

def imSummation (img1,img2,img_type = "8bit", imadjust = True) :  # save size  
    # input : 2 gray images
    # Output: 1 gray image
    imSum = np.array(np.add(img1, img2),dtype = float)
    imSum = imSum/ imSum.max()            # float
    if img_type is "16bit":
        imSum = skimage.img_as_uint(imSum)   # change to 16bit 
    else:
        imSum = skimage.img_as_ubyte(imSum)   # change to 16bit 

    if imadjust is True:
        p2, p98 = np.percentile(imSum, (2, 98))
        imSum = skimage.exposure.rescale_intensity(imSum, in_range=(p2, p98))

    return imSum

def imCompose (img1,img2, imadjust = True) :  # save size  
    print ("====imCompose==========")
    # input : 2 gray images
    # Output: 1 RGB image
    #    rgbArray = np.zeros((img1.shape[0],img1.shape[1],3))
    #    rgbArray[:,:, 0] = img1
    #    rgbArray[:,:, 1] = img2
    #    img_disp = Image.fromarray(rgbArray)        
    #    img_disp = cv2.merge((img1,img2,np.zeros_like(img1)))    
    img_disp = np.dstack((img1,img2,np.zeros_like(img1))) 
    if imadjust is True:
        p2, p98 = np.percentile(img_disp, (2, 99))
        img_disp = skimage.exposure.rescale_intensity(img_disp, in_range=(p2, p98))

    return img_disp

def image_prepare( write_folderName,  whole_image, masks_fName = "masks", whole_label =None, whole_class= None, crop_size = [300,300] , crop_overlap = 50 ,imadjust = 0, multithread = True,toRGB=False ,save_display =False):
    print ("*****image_prepare********")
    crop_width, crop_height    = crop_size
    img_rows = whole_image.shape[0]
    img_cols = whole_image.shape[1]                                                                      # img_rows = height , img_cols = width
    print ( "whole_image.shape=", whole_image.shape, "crop_size=", crop_size, 
            " crop_overlap=", crop_overlap, " imadjust=",  imadjust)
    class_names=["None","NeuN","IBA1","Olig2","S100","RECA1"] if whole_class is not None else ["None","Nucleus"]

    croppRange_ls = []
    for i in range(0, img_rows, crop_height - crop_overlap ):
       for j in range(0, img_cols, crop_width - crop_overlap):
           croppRange_ls.append ( [ i,j,crop_height + i, crop_width + j] )

    if toRGB is True:                                   #optional
        # trim the image to rgb
        if len(whole_image.shape) ==2:                  # gray
            whole_image = skimage.color.gray2rgb
        else:                                            #multiplex
            whole_image = whole_image[:,:,:3]

    if imadjust > 0 :
        CHN_img_ls = []
        for i in range(whole_image.shape[2]):
            CHN_img = whole_image [:,:,i]
            p_min, p_max = np.percentile(whole_image, ((100-imadjust), imadjust))
            CHN_img_ls. append( exposure.rescale_intensity(CHN_img, in_range=(p_min, p_max)) )           # Load over images
        whole_image = np.dstack (CHN_img_ls)

    def img_label_crop(croppRange_ls,masks_fName):
        for croppRange in croppRange_ls:
            imageID = str(croppRange[0]) + "_" + str(croppRange[1]) + "_adj" + str(imadjust)
            sample_folder    = os.path.join(write_folderName, imageID)
            if os.path.exists(sample_folder) is False:
                os.makedirs(sample_folder)       

            # Write Image, save as 1 .tif
            images_subfolder = os.path.join(sample_folder, "images")
            if os.path.exists(images_subfolder) is False:
                os.mkdir(images_subfolder)            
            cropped_image = dt_utils.cropping (whole_image        ,croppRange)
            cropped_image = dt_utils.zeropadding(cropped_image, canvas_size = crop_size) 
            if not os.path.exists( os.path.join (images_subfolder,imageID + ".tif") ) : 
                tifffile.imsave(os.path.join (images_subfolder,imageID + ".tif"), cropped_image.astype("uint8"))
            
            if whole_label is not None:
                cropped_label = dt_utils.cropping (whole_label    ,croppRange)       
            #    cropped_label = dt_utils.cleanEdgedComponents( dt_utils.cropping (label_array  ,croppRange) )                                        
                cropped_label = dt_utils.zeropadding(cropped_label, canvas_size = crop_size) 
                # Write Single label Imgs, save as several .png
                masks_subfolder  = os.path.join(sample_folder, masks_fName)    
                if os.path.exists(masks_subfolder) is False:
                    os.makedirs(masks_subfolder)
                mask_label_img = cropped_label[:,:]    
                
                numOfObjs = len(np.unique(mask_label_img) ) - 1                                         # exclude label =0
                print ("imageID=", imageID,"numOfObjs = ", str(numOfObjs))
                # put zeros for masks                                                                   #at least exist one mask
                # print ("Warning, their is no object detected in the image " ,
                #     PNG_img_FileName)
                m = np.zeros( ( cropped_image.shape[0],cropped_image.shape[1] ) , 
                                dtype = np.uint8)                                                       # incase there is no objs,save one empty

                class_ids = {}
                class_ids['id']=[]
                class_ids['class']=[]
                boxes_disp =[]
                masks_disp =[]
                obj_id = 0
                if numOfObjs > 0 :                                                                      # run only when there is at least one obj in the image 
                    labels = skimage.measure.label(mask_label_img)                                      # separate the object with same class ids
                    for obj in skimage.measure.regionprops(labels):
                        m = np.zeros( ( cropped_label.shape[0],cropped_label.shape[1] ) , 
                                    dtype = np.uint8)
                        m_filled_image = obj.filled_image*255
                        # pruning  of the labels                                                         # only write the object bboxsize  larger than 10 pixel                      # 
                        if  (obj.bbox[2] - obj.bbox[0] > 10) and \
                            (obj.bbox[3] - obj.bbox[1] > 10) : 
                                
                            m [ obj.bbox[0] : obj.bbox[2],
                                obj.bbox[1] : obj.bbox[3]] = m_filled_image
                                                        
                            skimage.io.imsave( os.path.join( masks_subfolder,
                                                            imageID + "-" + str (obj_id) + ".png"),
                                        m)
                            class_ids['id'].append(obj_id)                            
                            if whole_class is not None:   # multiclass
                                class_ids['class'].append(whole_class["class"][obj.label])                                
                            else:  # single class
                                class_ids['class'].append(1)                                
                           
                            if save_display is True:
                                boxes_disp.append(obj.bbox)
                                masks_disp.append(m)

                            obj_id = obj_id +1
                if numOfObjs ==0:  # put zeros for masks
                    print ("Warning, their is no object detected in the image " ,  imageID)
                    skimage.io.imsave( os.path.join( masks_subfolder,
                                                        imageID + "-0" + ".png"),
                                    m)
                    class_ids['id'].append(0)
                    class_ids['class'].append(0)
                    
                    if save_display is True:
                        boxes_disp.append([])
                        masks_disp.append(m)

                if whole_class is not None:
                    class_df =pd.DataFrame.from_dict(class_ids)
                    class_df.to_csv(os.path.join( masks_subfolder,
                                                  imageID + "-class" + ".csv"),
                                    index=False)
                    
                if save_display is True and  m.sum()!=0:

                    image_vis = cropped_image if len(cropped_image.shape)==2 else cropped_image[:,:,:3]
                    boxes       = np.array(boxes_disp)
                    masks       = np.stack(masks_disp, axis=-1)
                    
                    class_ids   = np.array(class_ids['class'],dtype=int)

                    print ("boxes.shape() = " , boxes.shape)
                    print ("masks.shape() = " , masks.shape)
                    print ("masks.shape[-1] = " , masks.shape[-1])
                    print ("class_ids = " , class_ids.shape)

                    #     assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

                    fig = display_instances( image = image_vis, boxes = boxes,   masks = masks,   
                                             class_ids = class_ids, class_names =class_names, mask_linewidth = 1,
                                             title= masks_fName +"_"+ imageID +"_annotation", show_bbox=False, show_mask=False)
                    fig. savefig(os.path.join( sample_folder,
                                                masks_fName +"_"+  imageID + "_annotation.png"))

    if multithread == False:  
        img_label_crop(croppRange_ls, masks_fName)
                                                                                                  # use single process function
    else:
        numOfThreads = multiprocessing.cpu_count()
        print ("numOfThreads = ", numOfThreads )
        pool = ThreadPool(processes=numOfThreads)
        ls_size = int(len(croppRange_ls)/numOfThreads)
        for th in range (0, numOfThreads):
            croppRange_ls_mp = croppRange_ls[  th*ls_size: th*ls_size +ls_size ]           
            pool.apply_async(img_label_crop(croppRange_ls_mp, masks_fName))
            print ("\tmulti thread for", th," ... \n")
    
    print (len(croppRange_ls), "Molded resuls have written in ",write_folderName)

def generate_validation(root_dir, val_rate = 0.05):
    import os, random, shutil
    #Prompting user to enter number of files to select randomly along with directory
    source=os.path.join(root_dir,"train")
    dest=os.path.join(root_dir,"val")
    if os.path.exists(dest) is False:
        os.makedirs(dest)       

    fns_datasets = os.listdir(os.path.join(source))    
    random.shuffle(fns_datasets)
    VAL_IMAGE_IDS  = fns_datasets[0:int(len(fns_datasets)*val_rate)]

    print("%"*25+"{ Details Of Transfer }"+"%"*25)
    print("\n\nList of Files Moved to %s :-"%(dest))

    #Using for loop to randomly choose multiple files
    for VAL_IMAGE_ID in VAL_IMAGE_IDS:
        #Variable random_file stores the name of the random file chosen
                
        source_file = os.path.join(source,VAL_IMAGE_ID)
        dest_file   = os.path.join(dest  ,VAL_IMAGE_ID)
        #"shutil.move" function moves file from one directory to another
        shutil.move(source_file,dest_file)

    print("\n\n"+"$"*33+"[ Files Moved Successfully ]"+"$"*33)

def str2bool(str_input):
    bool_result = True if str_input.lower() in ["t",'true','1',"yes",'y'] else False
    return bool_result

if __name__ == '__main__':
    import argparse,time
#     import matplotlib
# #     # Agg backend runs without a display
#     matplotlib.use('Agg')
#     import matplotlib.pyplot as plt

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='****  Whole brain segentation pipeline on DAPI + Histone Channel ******' +
        '\ne.g\n' +
        '\t$ python3 registrationORB_RANSIC.py  # run on whole brain \n ' +
        '\t$ python3 registrationORB_RANSIC.py -t T  ' +
        '-i /data/jjahanip/50_plex/stitched  ' +
        '-w /data/xiaoyang/CHN50/Registered_Rebecca',
        formatter_class=argparse.RawTextHelpFormatter)
        
    parser.add_argument("command",
                        metavar="<command>", default = 'detect',
                        help="'train' or 'detect'or 'val'")       
    parser.add_argument('-i','--img', required=False,
                        metavar = "/path/to/dataset/",
                        default = None,
                        help='Root directory of images to be registerd, default = "/data/jjahanip/50_plex/stitched"')                
    parser.add_argument('-l','--label', required=False,
                        metavar="/path/to/label/",default = None,
                        help='label for whole to train MRCNN')   
    parser.add_argument('--class_df', required=False,default= None,
                        metavar="/path/to/class/",
                        help='class table for label image,1st column: label_id,2nd column: class id')                           
    parser.add_argument('--imadjust', required=False,
                        default = 0, type = int, 
                        help='image adjust rate[ 0: not adjust, 100:equalscale(0,100), 98: equal_rescale to (2,98) ')                            
    parser.add_argument('--toRGB', required=False,
                        default = '0', type = str, 
                        help='whether to convert the image to RGB channel') 
    parser.add_argument('--save_display', required=False,
                        default = '0', type = str, 
                        help='whether to save the display mask images for each samples')
    parser.add_argument('--val', required=False,
                        default = None, type = str, 
                        help='validation set rate to training set( e.g: 0.05)') 
    parser.add_argument('--masks_fName', required=False,
                        default = "masks", type = str, 
                        help='masks subfoler name to storage the masks') 
    parser.add_argument('--multiprocess', required=False,
                        default = '1', type = str, 
                        help='whether to conduct multiprocessing') 
    parser.add_argument('-r','--results', required=False,
                        metavar="/path/to/result/",default = None,
                        help='directory for results, default = ROOT_DIR+"results/nucleus/") ')
    args = parser.parse_args()

    tic = time.time()

    if args.command != "val":
        wholeImage = tifffile.imread(args.img).astype(np.uint8)
        # if str2bool(args.toRGBOpt) == True:                                   #optional
        #     # trim the image to rgb
        #     if len(wholeImage.shape) ==2:                                   # gray to rgb
        #         wholeImage = skimage.color.gray2rgb
        #     else:                                                            # multiplex  to rgb
        #         wholeImage = wholeImage[:,:,:3]

        print ("Read wholeImage.shape=",wholeImage.shape )
    
    # Create model
    if args.command == "train":
        if args.label is not None:
            if ".npy" in args.label:
                wholelabel = np.load( args.label )
            else:
                wholelabel = np.loadtxt( args.label ,delimiter=",", dtype = int)
        else:
            wholelabel = None
        output_dir = os.path.join(args.results,args.command)   # output_dir = args.results/train
        if os.path.exists(output_dir) is False:
            os.mkdir(output_dir)
            
        if args.class_df is not None:
            wholeclass = pd.read_csv(args.class_df)
            if "avg_uniqueType" in list(wholeclass.columns):
                print ("load class from avg_uniqueType")
                wholeclass["class"] = wholeclass[class_key]
            else:
                temp_array = np.array( wholeclass[["NeuN", "Iba1", "Olig2", "S100","RECA1"]])
                wholeclass["class"] = temp_array.argmax(axis=1) +1
                del temp_array
        else:
            wholeclass = None        

        image_prepare( output_dir,  masks_fName = args.masks_fName,
                        whole_image  = wholeImage, whole_label =wholelabel, whole_class= wholeclass,
                        crop_size = [512,512] , crop_overlap = 0, imadjust = args.imadjust,    
                        multithread     = str2bool(args.multiprocess),
                        toRGB           = str2bool(args.toRGB) ,
                        save_display    = str2bool(args.save_display)  )

    elif args.command == "val":
        root_dir = args.results
        if args.val is not None :
            generate_validation(root_dir, val_rate = float(args.val ))

    else: # testing
        image_prepare( args.results,  whole_image  = wholeImage,
                        crop_size = [512,512] , crop_overlap = 50, imadjust = int(args.imadjust),     
                        multithread     = str2bool(args.multiprocess),
                        toRGB           = str2bool(args.toRGB) ,
                        save_display    = str2bool(args.save_display)  )

    toc2 =  time.time()
    print ("total time = ", int(toc2 - tic))