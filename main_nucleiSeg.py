
'''
Run whobrain detection
e.g
label_dir="/brazos/roysam/xli63/exps/Data/50_plex/jj_final/autoseg_seedsProvides/DPH"
nohup python3 nucleus_wholebrain_train_detect_merge.py train \
--dataset=../../datasets/CHN50_mold/RDGHBDPH.tif \
--label="$label_dir"/[DAPI+Histones]219634seeds_fill_Falselabels_nuclear.npy  \
--weights=imagenet \
--logs=../../results/[train]-"$dName"-molded-train--auto-"$preweight" \
2>&1 | tee logs/log[train]_img-RDGHBDPH_label-auto_weights-imagenet.txt
python3 nucleus_wholebrain_train_detect_merge.py train \
--dataset=/brazos/roysam/xli63/exps/Data/50_plex/jj_final/images_stacked_multiplex/multiplex.tif \
--label=../../datasets/CHN50_mold/[DAPI+Histones]autoseg_208835seeds_label.out \
--weights=imagenet
'''
# Set matplotlib backend
# See more at Training with RGB-D or Grayscale images
# https://github.com/matterport/Mask_RCNN/wiki


# change the loading method , no need to create the cropped images, just whole brain as input
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
import sys
import json
import datetime
import numpy as np
if "1.18" in np.version.version:
    np.random.bit_generator = np.random._bit_generator   #  error introduced by numpy 1.18 
# import tifffile as tiff 
import skimage
from skimage import exposure, segmentation,io,img_as_ubyte,measure,morphology
import skimage.external.tifffile as tiff 
import cv2
from imgaug import augmenters as iaa
import warnings
import h5py
warnings.filterwarnings("ignore")


datauti_libPath = os.path.join( os.path.dirname(os.path.realpath(__file__)),"NUCLEAR_SEG/supplement")
sys.path.insert(0, datauti_libPath)
# import pdb; pdb.set_trace()
import merge_detection_result as merge_crop_fcts
import datasets_utils as dt_utils


ROOT_DIR = os.path.abspath("../../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
# from mrcnn import model as modellib

model_libPath =  os.path.join( os.path.dirname(os.path.realpath(__file__)), 'NUCLEAR_SEG/mrcnn')
sys.path.insert(0, model_libPath)

import tensorflow as tf
from pkg_resources import parse_version
if parse_version(tf.__version__) >= parse_version('2'):
    import model_tf2 as modellib
else:
    import model_tf1 as modellib

from mrcnn import utils 
from mrcnn import visualize
import pandas as pd
import scipy
import random
import argparse,time

# avoid showing warning
from tensorflow.python import util
util.deprecation._PRINT_DEPRECATION_WARNINGS = False

os.environ["CUDA_VISIBLE_DEVICES"] = "0"




#======================================================
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

class NucleusConfig(Config):

    def __init__(self, dataset):
        """Set values of computed attributes."""
        # Effective batch size
        self.IMAGE_CHANNEL_COUNT = 3
        self.set_IMAGE_CHANNEL_COUNT(self.IMAGE_CHANNEL_COUNT)
        # Image meta data length
        # See compose_image_meta() for details
        self.dataset = dataset      
        self.KEEP_PROB = 1
        self.NUM_CLASSES = 2
        self.crop_size = [512,512]
        self.set_NUM_CLASSES(self.NUM_CLASSES)
        self.set_IMAGES_PER_GPU(self.IMAGES_PER_GPU)
        
    def set_NUM_CLASSES(self,NUM_CLASSES):
        self.NUM_CLASSES = NUM_CLASSES
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

    def set_crop_size     (self, crop_size):
        self.crop_size = crop_size        

    def set_IMAGES_PER_GPU  (self, IMAGES_PER_GPU):                     #                     15
        self.IMAGES_PER_GPU = IMAGES_PER_GPU
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT          #  #                     15
    
    def set_IMAGE_CHANNEL_COUNT(self, image_channel_count):
        self.IMAGE_CHANNEL_COUNT = image_channel_count
        # Input image size
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM, self.IMAGE_CHANNEL_COUNT])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, self.IMAGE_CHANNEL_COUNT])

        if self.IMAGE_CHANNEL_COUNT ==3:
            # self.MEAN_PIXEL = np.array([43.53, 39.56, 48.22])                            # # Image mean (RGB)
            self.MEAN_PIXEL = np.array([45,45,45])                            # # Image mean (RGB)

        else:
            MEAN_PIXEL_value = np.array([45])
            self.MEAN_PIXEL = np.tile(MEAN_PIXEL_value,self.IMAGE_CHANNEL_COUNT)          ## Image mean (multiplex) ##


    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "nucleus"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 15                                                  #     IMAGES_PER_GPU = 6   

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + nucleus

    VAL_IMAGE_IDS = []


    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = (657 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU
    VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0.5

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0                   # change to multiplex 

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9                                         

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    # MEAN_PIXEL = np.array([43.53, 39.56, 48.22])                  #RGB
    # MEAN_PIXEL = 45                                              ## multiplex ##

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 500                                   # origninally 100

class NucleusInferenceConfig(NucleusConfig):
    
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    # GPU_COUNT = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.90  # raw == 0.7
    DETECTION_MIN_CONFIDENCE = 0.1   # lower than this will not show more  # orignial 0.7

############################################################
#  Dataset
############################################################


class WholeBrainDataset(utils.Dataset):
    
    def set_DIRorIMG(self,dataset_dir):
        self.DIRorIMG = "DIR" if os.path.isdir(dataset_dir) is True else "IMG"
    def set_toRGB(self,toRGB):
        self.toRGB = toRGB
    def set_masks_fName(self,masks_fName = "masks"):
        self.masks_fName = masks_fName
    def set_keyImgID(self,keyImgID = None):
        self.keyImgID = keyImgID

    #############  dataset_dir is wholeImage ##################
    def load_wholeImage(self, wholeImage_memmap, crop_size = [512,512],  crop_overlap = 50, save_path = None ):
        """Load a the nuclei dataset.
        dataset_dir: Root directory of the dataset        
        crop_overlap=50 for testing/detect
        crop_overlap=0 for training
        """
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("nucleus", 1, "nucleus")
        self.wholeImage_memmap = wholeImage_memmap

        # dataset dir has to be the whole image file 
        try:
            # img = tiff.imread(dataset_dir)  # change for [multiplex]
            # print ("img.shape=", img.shape)
            img_rows = wholeImage_memmap.shape[0]
            img_cols = wholeImage_memmap.shape[1]
            # del img
        except OSError:
            print('dataset_dir is not a image full path',)

        # Prepare crop image ids and  Add images        
        crop_width, crop_height    = crop_size
        print ("#"*10, "Start preparing the images" )
        print ("crop_size = ", crop_size ,"crop_overlap = ",crop_overlap )
        image_counter = 0 
        for i in range(0, img_rows, crop_height - crop_overlap ):
            for j in range(0, img_cols, crop_width - crop_overlap):
                image_id = str(i) + "_" + str(j)            # image id define as xmin,ymin
                image_counter = image_counter + 1 
                if save_path is None:                           # note image crop as the whole image path
                    self.add_image(
                        "nucleus",
                        image_id=image_id,
                        path="")
                else: # don't need to use it only for sanity check
                    cropped_img = self.load_image_crop( image_id,wholeImage_memmap, crop_size = [512,512])
                    cropped_img_path = os.path.join(save_path,image_id + ".tif")
                    # tiff.imwrite (cropped_img_path,cropped_img)
                    tiff.imsave (cropped_img_path,cropped_img)

                    self.add_image(
                        "nucleus",
                        image_id=image_id,
                        path=cropped_img_path)
        print ("Last image_id= ",image_id , " in total image number is ", image_counter)
    
    def load_mask_crop(self, image_id, crop_size = [512,512]):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        i = int( image_id.split("_")[0] ) 
        j = int( image_id.split("_")[1] ) 

        crop_width, crop_height    = crop_size
        croppRange =  [ i,j,crop_height + i, crop_width + j]
        cropped_label = dt_utils.cropping   (wholelabel   , croppRange)
        cropped_label = dt_utils.zeropadding(cropped_label, canvas_size = crop_size) 
        cropped_label = cropped_label.astype(wholelabel.dtype)

        mask = []
        # print ("\n&&&&&&&&&&&&&&&&loading mask for image_id=",image_id)
        for obj in skimage.measure.regionprops (cropped_label):
            m = np.zeros(crop_size,dtype=np.bool)
            m [ obj.bbox[0]:obj.bbox[2],  obj.bbox[1]:obj.bbox[3]] = obj.filled_image
            mask.append(m.astype(np.bool))

        if mask == []:                                                  # no cell has been detected
            mask.append(np.zeros(crop_size,dtype=np.bool))
        mask = np.stack(mask, axis=-1)

        return mask, np.ones([mask.shape[-1]], dtype=np.int32)
                
    def load_image_crop(self, image_id, wholeImage_memmap,crop_size = [512,512]):
        # change wholeImage to global variable
        """Load the specified image and return a [H,W,D] Numpy array.
        """
        
        i = int (image_id.split("_")[0])
        j = int (image_id.split("_")[1])

        crop_width, crop_height    = crop_size
        croppRange =  [ i,j,crop_height + i, crop_width + j]
        cropped_image = dt_utils.cropping   (wholeImage_memmap   , croppRange)
        cropped_image = dt_utils.zeropadding(cropped_image, canvas_size = crop_size) .astype(wholeImage_memmap.dtype)

        image = cropped_image
        # If grayscale. Convert to RGB for consistency.
        if image.ndim == 2:
            image = skimage.color.gray2rgb(image)
        # # If has an alpha channel, remove it for consistency
        # if image.shape[-1] == 4:
        #     image = image[..., :3]            
        # If has an alpha channel, remove it for consistency
        if image.ndim == 4:
            image = image[..., :3]      
        if self.toRGB== True:
            image = image[:,:,:3]
        return image

    def del_whole_data(self):
        if "wholelabel" in globals():
            del globals()["wholelabel"]
        if "wholeImage"in globals():
            del globals()["wholeImage"]
   
    #############  dataset_dir is cropped samples ##################
    def load_dataset(self, dataset_dir, keyImgID = ""):
        """Load a subset of the nuclei dataset.
        dataset_dir: Root directory of the dataset
        keyImgID: only load the image id  if it contain "keyImgID"
        """
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("nucleus", 1, "nucleus")
        if ".tif" in os.listdir(dataset_dir)[0]:            # load image directly in the directory
            for image_id in os.listdir(dataset_dir):
                if keyImgID in image_id:
                    # print ("...load sample ",image_id)
                    self.add_image(
                        "nucleus",
                        image_id=image_id,
                        path=os.path.join(dataset_dir, image_id ) )
        else:                                           # load image in "/images" foler
            image_ids = next(os.walk(dataset_dir))[1]            
            # Add images
            image_ext = os.listdir(os.path.join(dataset_dir, image_ids[0], "images"))[0].split(".")[1]
            for image_id in image_ids:
                if keyImgID in image_id:
                    # print ("...load sample ",image_id)
                    self.add_image(
                        "nucleus",
                        image_id=image_id,
                        path=os.path.join(dataset_dir, image_id, "images/{}.".format(image_id) + image_ext))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        # mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), self.masks_fName)
        ### save memory for *_adj98, look for the mask in _adj0
        if os.path.exists(mask_dir) is False:
            mask_dir = os.path.join( mask_dir.split("_adj")[-2] + "_adj0", self.masks_fName)

        # Read mask files from .png image
        mask = []
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith(".png"):
                m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
                mask.append(m)
        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)
    
    
    ############ Common ##############
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "nucleus":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)        

    def load_image(self, image_id):  # over-writed from util
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image

        image = tiff.imread(self.image_info[image_id]['path'])

        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.ndim == 4:
            image = image[..., :3]      
        if self.toRGB== True:
            image = image[:,:,:3]
        return image
############################################################
#  Training
############################################################

def train_model(model, dataset_dir,crop_size =[512,512] , 
        augmentation = True, masks_fName = "masks",keyImgID = "",toRGB = False, epoch = 40):

    """Train the model."""
    # Training dataset.
    dataset_train = WholeBrainDataset()
    dataset_train.set_DIRorIMG(dataset_dir)             # define whether dataset_dir is a dir or file    dataset_train.set_masks_fName(masks_fName)    dataset_train.set_masks_fName(masks_fName)
    dataset_train.set_masks_fName(masks_fName)
    dataset_train.set_toRGB(toRGB)
    dataset_train.set_keyImgID(keyImgID)

    if dataset_train. DIRorIMG == "DIR": 
        dataset_train.load_dataset(dataset_dir , keyImgID)
    else:        
        with tiff.TiffFile(dataset_dir) as tif:
            wholeImage_memmap = tif.asarray(memmap=True)
            wholeImage_memmap = wholeImage_memmap[:,:int( wholeImage_memmap.shape[1]/2 ),:]   # only train the left half
        dataset_train.load_wholeImage(wholeImage_memmap,crop_size = crop_size,  crop_overlap = 0 )

    dataset_train.prepare()

    # Validation dataset 
    prospective_VAL_DIR = os.path.join( os.path.dirname(dataset_dir), "val")
    if os.path.exists(prospective_VAL_DIR ):  # already have val set
        dataset_val = WholeBrainDataset()
        dataset_val.set_DIRorIMG(prospective_VAL_DIR)             # define whether dataset_dir is a dir or file
        dataset_val.set_masks_fName(masks_fName)
        dataset_val.set_keyImgID(keyImgID)
        dataset_val.set_toRGB(toRGB)

        if dataset_val. DIRorIMG == "DIR": 
            dataset_val.load_dataset(prospective_VAL_DIR)
        else:
            dataset_val.load_wholeImage(prospective_VAL_DIR,crop_size = crop_size,  crop_overlap = 0 )
        dataset_val.prepare()
    else:
        dataset_val = dataset_train

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    if augmentation is True:
        # augmentation = iaa.SomeOf((0, 2), [
        #     # iaa.Fliplr(0.5),
        #     # iaa.Flipud(0.5),
        #     iaa.OneOf([iaa.Affine(rotate=90),
        #                 iaa.Affine(rotate=180),
        #                 iaa.Affine(rotate=270)]),
        #     iaa.Multiply((0.8, 1.5)),
        #     iaa.GaussianBlur(sigma=(0.0, 5.0)),
        #     # add below
        #     iaa.GammaContrast((0.5, 2.0), per_channel=True)
        # ])
        #     # https://imgaug.readthedocs.io/en/latest/source/overview/contrast.html

        augmentation = iaa.SomeOf((0, 2), [                      # only pick 0~2 of the aug, avioid redundant/ toostrong aug
            iaa.GammaContrast((0.5, 1.2), per_channel=True)   ,  # Change color/ switch channels
            iaa.ChangeColorspace( "GRAY"  ),
            iaa.LinearContrast((0.4, 1.6), per_channel=True) ,    #  Strengthen or weaken the contrast in each image.

            iaa.SigmoidContrast(                                    # Adjust image contrast
                    gain=(3, 10), cutoff=(0.4, 0.6), per_channel=True) ,     
            iaa.OneOf([
                        iaa.Affine(rotate=90),
                        iaa.Affine(rotate=180),
                        iaa.Affine(rotate=270),
                        ]),
            iaa.OneOf([                        
                        iaa.Multiply((0.8, 1.5)),
                        iaa.GaussianBlur(sigma=(0.0, 5.0)),                
                        
                        ]),          
        ])

    else:
        augmentation = None
    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    # print ("augmentation = ", augmentation)
    # print("Train network heads")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=20,
    #             augmentation=augmentation,
    #             layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=epoch,
                augmentation=augmentation,
                layers='all')

############################################################
#  Detection
############################################################


def detect_model(model, dataset_dir, crop_size =[512,512],batch_size = 1, result_dir = None,
        imadjust = False ,uncertainty=False, toRGB = False, bgBoost = 1, savefig_dir = None,verbose=0):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory       
    # crop_imgs_dir = os.path.join(result_dir,"cropped_imgs")
    # os.makedirs(crop_imgs_dir, exist_ok=True)
    t_0 = time.time()
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(result_dir, submit_dir)
    
    os.makedirs(submit_dir)
    submission = [] 
    score = {}
    # featureTable = pd.DataFrame(   [list(np.zeros(8))],
    #                                 columns = [ "cropimg_ID","score", "centroid_x" , "centroid_y", "min_row", "min_col", "max_row", "max_col"] )

    print("start to write Saved to ", submit_dir)
    print ("bgBoost=",bgBoost)
    # Read dataset for detection 
    dataset_test = WholeBrainDataset()
    dataset_test.set_DIRorIMG(dataset_dir)                                   # define whether dataset_dir is a dir or file
    dataset_test.set_toRGB(toRGB)
    submission = []
    imadjust_vis = True
    BATCH_SIZE = batch_size
    t_d = time.time()
    

    ##### Prepare load testing set
    if dataset_test. DIRorIMG == "DIR":   # image cropps
        dataset_test.load_dataset(dataset_dir)  # crop_size = 512
        dataset_test.prepare()
        # Load over images

    else: #whole image, need to crop it  and save them into crop_imgs_dir
        with tiff.TiffFile(dataset_dir) as tif:
            wholeImage_memmap = tif.asarray(memmap=True)
        # dataset_test.load_wholeImage(wholeImage_memmap,crop_size = crop_size ,  
        #                             crop_overlap = 50 ,save_path = crop_imgs_dir)  # crop_size = 512        
        dataset_test.load_wholeImage(wholeImage_memmap,crop_size = crop_size ,  
                                    crop_overlap = 50 )  # crop_size = 512
        dataset_test.prepare()
        dataset_test.del_whole_data()

    ##### start testing
    if BATCH_SIZE > 1:                                                          # batch testing  only background boosting once      
       
        for i, image_id  in enumerate( dataset_test.image_ids ) :
            # Load image and run detection
            if i % BATCH_SIZE == 0:                                             #initalize batch
                batch_image_name = []
                batch_image = []
                batch_bg_image = []            

            image_name =    dataset_test.image_info[image_id]["id"] 
            # image = dataset_test.load_image (image_name)         # [multiplex]   # crop_size = 512
            image = ( dataset_test.load_image_crop ( image_name,wholeImage_memmap, crop_size = [512,512]) 
                     if dataset_test. DIRorIMG == "IMG"  else   dataset_test.load_image(image_id) )
            if imadjust == True:
                image = dt_utils.image_adjust(image)                        # Load over images
            
            batch_image_name.append(image_name)
            batch_image.append(image)
            
            if i % BATCH_SIZE == BATCH_SIZE -1 or i== len(dataset_test.image_ids) -1:     # batch assembling done
                # Detect objects
                t_d = time.time()
                if len (batch_image) < BATCH_SIZE:  # add zeros image to fix the batch size
                    batch_image = batch_image + [np.zeros_like(image)]* ( BATCH_SIZE-len (batch_image) )
                
                # import pdb; pdb.set_trace()
                batch_r = model.detect(batch_image, verbose=0)       
                for r_i , image_name in enumerate ( batch_image_name):  # r_i <= BATCH_SIZE
                    r , image = batch_r[r_i], batch_image[r_i]
                    rle = dt_utils.maskScoreClass_to_rle(image_name, r["masks"], r["scores"],r['class_ids'] )       
                    submission.append(rle)    
                    # prepard background image   
                    if bgBoost > 0:
                        mask_bg = dt_utils.mask_bg(r["masks"])   # get the undetected pixel [0: bg, 1: masked]                    
                        mask_bg = morphology.binary_erosion(mask_bg, morphology.disk(1))   # make mask thinner
                        mask_bg = np.dstack([mask_bg]*image.shape[2])

                        bg_image = dt_utils.image_adjust(image,80)*(mask_bg==0 )     # bg on adjusted image                        
                        batch_bg_image .append(bg_image)

                if bgBoost > 0:            
                    # Detect background objects
                    if len (batch_bg_image) < BATCH_SIZE:                            # add zeros image to fix the batch size
                        batch_bg_image = batch_bg_image + [np.zeros_like(image)]* ( BATCH_SIZE-len (batch_bg_image) )

                    batch_r_bg = model.detect(batch_bg_image, verbose=0)             # bgBoost to the background
                    for r_i , image_name in enumerate ( batch_image_name):           # r_i <= BATCH_SIZE
                        r_bg , image = batch_r_bg[r_i], batch_image[r_i]
                        # Encode image to RLE. Returns a string of multiple lines
                        rle_bg = dt_utils.maskScoreClass_to_rle(image_name, r_bg["masks"], r_bg["scores"],r_bg['class_ids']) 
                        submission.append(rle_bg)     
                    
    else:                                                                       # image testing, batch size =1 , background boosting  until no objs
        numOfBgCells_tol =0
        MAX_ITER = int(bgBoost)                                            # approximate to the maximun number of cells in the object
        print("Background boosting(bgBoost):MAX_ITER=",MAX_ITER)
        bgBoost_ls = []
        image_name_ls=[]
        r_masks =[]
        for i, image_id  in enumerate( dataset_test.image_ids ) :
            # Load image and run detection
            image_name = dataset_test.image_info[image_id]["id"]
            t_0 = time.time()            
            submission_local = []
            # Raw Image
            image = ( dataset_test.load_image_crop ( image_name,wholeImage_memmap, crop_size = [512,512]) 
                     if dataset_test. DIRorIMG == "IMG"      else   dataset_test.load_image(image_id) ) 
            
            image_name = image_name.split(".")[0] if "." in image_name else image_name   # if not crop,might contrain .tif

            image_raw = image.copy()
            if imadjust == True:
                image = dt_utils.image_adjust(image)                            # Load over images

            r = model.detect([image], verbose=0)[0]                
            rle = dt_utils.maskScoreClass_to_rle(image_name, r["masks"], r["scores"],r['class_ids'] )       
            submission.append(rle)
            submission_local.append(rle)     
            r_masks = r["masks"]                                                # the fg masks
            bg_labels = dt_utils.masks_to_label(r_masks)                        # Initalize
            mask_bg_delta = 0
            mask_bg_previous = np.array(0)
            bg_it = {}
            bg_it["img_id"] =[image_name]

            if bgBoost > 0:                                                     #  BG boosting 
                     
                for it in range( 0,MAX_ITER):
                    if it ==0:
                        numOfBgCells = r["masks"].shape[2] 

                    else:                                              
                        mask_bg = dt_utils.mask_bg(r_masks)                      # get the undetected pixel [0: bg, 1: masked]                    
                        mask_bg_delta = mask_bg.sum() - mask_bg_previous.sum()
                        # mask_bg = morphology.binary_erosion(mask_bg, morphology.disk(1))   # make mask thinner
                        # bg_image = dt_utils.image_adjust(image,80)*(mask_bg==0 )  # bg on adjusted image                        
                        image = image *(np.dstack([mask_bg]*image.shape[2])==0 ) .copy()                        # update the new image with the bg from the previous                        
                        r = model.detect([image], verbose=0)[0]                
                        numOfBgCells = r["masks"].shape[2]                              # the number of cells detected in the BG    
                        if numOfBgCells ==0 :
                            break                                                   # until no cells in the background image
                        else:
                            # import pdb;pdb.set_trace()
                            r_masks = np.concatenate ([r_masks,r["masks"]],axis=2)
                            rle = dt_utils.maskScoreClass_to_rle(image_name, r["masks"], r["scores"],r['class_ids'] )       
                            submission.append(rle)
                            submission_local.append(rle)     
                            numOfBgCells_tol += numOfBgCells
                            mask_bg_previous = mask_bg                           

                            if  mask_bg_delta == 0:
                                break
                    if verbose and savefig_dir is not None:
                        print (image_name+"-it:"+str(it)," detect numOfBgCells:",numOfBgCells)       #save intermediate result for each iterations
                        fg_label = dt_utils.masks_to_label(r["masks"])                                        
                        label = dt_utils.masks_to_label(r_masks)
                        dt_utils.bgBoost_savevis(img      = image_raw,  labels=label,  fglabels = fg_label,bglabels= bg_labels,
                                                img_id   = image_name+"-it"+str(it),   output_dir = savefig_dir,
                                                imadjust = True,                )

                        bg_labels = dt_utils.masks_to_label(r_masks)            # current label is the bg label for next round

                        bg_it[it] = [numOfBgCells]

            if  dataset_test. DIRorIMG == "DIR":                         # write the annoation results into "pred_val"
                submission_local = "ImageId,EncodedPixels,Score,Class\n" + "\n".join(submission_local)
                file_path_local = os.path.join(submit_dir, "submission_local.csv")
                with open(file_path_local, "w") as f:
                    f.write(submission_local)
                relabels = merge_crop_fcts.result_merge(image, file_path_local,local=True)                   # check overlap
                
                dt_utils.label_savevis(img      = image_raw,    label = relabels,
                                       img_id   = image_name,   output_dir = savefig_dir,
                                       imadjust = True     )

            if verbose:
                print ("\t Detect image",image_name,"used (s)=",   str( time.time()  -  t_0))    
                # import pdb;pdb.set_trace()

                bg_it = pd.DataFrame.from_dict(bg_it)        
                bgBoost_ls.append(bg_it)
                # import pdb;pdb.set_trace()

        if verbose:
            if  dataset_test. DIRorIMG == "DIR" and bgBoost > 0:        # parameter tunning, record how many iterations
                df =  pd.concat(bgBoost_ls, axis=0, sort=False)
                df.to_csv(os.path.join(savefig_dir,"bgBoost.csv"))
        print("Background boosting detected cells:",numOfBgCells_tol)
                    
    print ("Detect image used average time (s)=",   str( ( time.time()  -  t_d) /i)  )            
    dataset_test.del_whole_data()                                                           # free space
    # Save to csv file
    submission = "ImageId,EncodedPixels,Score,Class\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")

    with open(file_path, "w") as f:
        f.write(submission)
    print("\n Raw rles of cropped images Saved to ", file_path)
    # featureTable.to_csv( os.path.join(submit_dir, "FeatureTable.csv"))

    return file_path


############################################################
#  Command Line
############################################################
def str2bool(str_input):
    bool_result = False if str_input.lower() in ["f",'false','0',"no",'n'] else True
    return bool_result

def get_image_shape(dataset_dir):

    demo_image_path = os.path.join(dataset_dir, os.listdir(dataset_dir)[0],"images")
    if os.path.exists(demo_image_path):  # if is a folder
        demo_image_path = os.path.join( demo_image_path, os.listdir(demo_image_path)[0] )
    else:                          
        demo_image_path = os.path.join(dataset_dir, os.listdir(dataset_dir)[0])
    # import pdb;pdb.set_trace()
    demo_image_shape = io.imread(demo_image_path).shape
    if len(demo_image_shape) ==3:
        image_channel_count = demo_image_shape[2]
    else:
        image_channel_count = 1
    print("image_channel_count =!!!", image_channel_count)  
    return demo_image_shape[0], demo_image_shape[1], image_channel_count,

def load_latest_weight(weight_root):
    weights_path ,latest_weight_path = None,None
    if os.path.isdir(weight_root):  # is a dir, then search for the lasted epoch of all the weights in the folder 
        epoch_max=0
        for subfolder in os.listdir(weight_root):
            if "nucleus" in subfolder:
                ## tf2version, the h5fild is under "train"
                # if os.listdir(os.path.join(weight_root,subfolder))[0] == "train":
                #     subfolder =  subfolder+"/train" 
                for h5file in os.listdir(os.path.join(weight_root,subfolder)):
                    if ".h5" in h5file:
                        temp = os.path.join(weight_root,subfolder,h5file)
                        epoch = int( h5file.split("_0")[-1].split(".h5")[0])
                        if epoch > epoch_max:
                            latest_weight_path = temp
                            epoch_max = epoch
        weights_path = latest_weight_path
    return weights_path

if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for nuclei counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>", default = 'detect',
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/dataset/",
                        default = "/project/ece/roysam/xiaoyang/exps/Data/50_plex/jj_final/atlas/multiplex_atlas",
                        help='Whole image to apply cell segmentation')           
    parser.add_argument('--imadjust', required=False,
                        default = "F",type = str, 
                        help='whether to adjust the image, or the perc to apply adjust e.g. no adjust:"0", adjust(default perc 98): "1", "70"')         
    parser.add_argument('--toRGBOpt', required=False,
                        default = '0',type = str, 
                        help="0 false(load all channels) ,1 true (load only the first 3 channels)")                    
    parser.add_argument('--weights', required=False,
                        default="NUCLEAR_SEG/weights/mrcnn_weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    args, _ = parser.parse_known_args()

    if args.command == 'train' or args.command == 'retrain':
        parser.add_argument('--label', required=False,
                        metavar="/path/to/label/",default= None,
                        help='label for whole to train MRCNN')                           
        parser.add_argument('--logs', required=False,
                        metavar="[train]/path/to/logs_for training weights/",
                        help='Logs and checkpoints directory (default=logs/)')
        parser.add_argument('--epoch', required=False,
                        default= 40, help='total number of training epoch, default=40')
        parser.add_argument('--keyImgID', required=False,
                            metavar="[train/detect]sample key work name",default= "",
                            help='sample name keyword')       
        parser.add_argument('--masks_fName', required=False,
                            metavar="[detect] mask subfolder name",default= "masks",
                            help='subfolder name of masks')      
        parser.add_argument("--fgmask" ,required=False,
                            default = None,type = str, 
                            help="the path of the foreground mask of the image (for nuclei stained channel as input)")   

        args, _ = parser.parse_known_args()                

    elif args.command == 'detect' or args.command =='val':
        parser.add_argument('--results', required=False,
                        metavar="[detect]/path/to/detection_result/",default = None,
                        help='directory for results, default = ROOT_DIR+"results/nucleus/") ')
        parser.add_argument('-bb','--bgBoost', required=False, 
                            default = '20',type = int, 
                            help="background boosting max iterations")   
        parser.add_argument('-bz','--batch_size', required=False,
                            default = '1',type = str, 
                            help= "batch size for testing")    
        parser.add_argument('--val_crops', required=False,default=None,
                        metavar="/path/to/crop_img_id.csv/",
                        help='the path to load the crop_img id for savinng the detection result for validation')   
        parser.add_argument('--visimg', required=False,default=None,
                            metavar="/path/to/visimg/",
                            help='the image to superimpose detection result on')   
        args, _ = parser.parse_known_args()                         

    args = parser.parse_args()

    tic = time.time()

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)

    # Results directory
    # Save submission files here
    if args.command == "detect":
        os.makedirs(args.results, exist_ok=True)

    ##########  Load image #################

    if os.path.isdir(args.dataset) is False :  # Load images path
        # global wholeImage
        print ("loading whole brain image:", args.dataset)

        # whole image preprocessing    # not working ??
        with tiff.TiffFile(args.dataset) as tif:
            rawImage = tif.asarray(memmap=True)

        # import pdb;pdb.set_trace()
        # rawImage = tiff.imread(args.dataset)
        if str2bool(args.imadjust):   # True
            if args.imadjust.isdigit() is False:
                wholeImage = dt_utils.image_adjust(rawImage)                     # Load over images
            else:
                if int(args.imadjust) ==1:  # 1 
                    wholeImage = dt_utils.image_adjust(rawImage)                 # Load over images
                else:
                    wholeImage = dt_utils.image_adjust(rawImage,
                                                int (args.imadjust))             # adjust a certain perc, e.g.70
        else:
            wholeImage = rawImage

        if rawImage.dtype == np.uint16:
            wholeImage = skimage.img_as_ubyte(wholeImage )                       # read 16 bit gray scale image.
        else:
            wholeImage = wholeImage
        
        if str2bool(args.toRGBOpt) :                                             # optional
            # trim the image to rgb
            if  len(rawImage.shape) ==2 :                                        # gray to rgb
                wholeImage = skimage.color.gray2rgb(wholeImage).copy()
            else:                                                                # multiplex  to rgb
                wholeImage = wholeImage[:,:,:3].copy() 
        else:
            wholeImage = wholeImage.copy()

        wholeImage_shape = wholeImage.shape
        print ("wholeImage=:", wholeImage.max(),wholeImage.dtype)
        image_channel_count  = wholeImage.shape[2]
        wholeImage = wholeImage.astype(np.uint8)                                  # read 16 bit gray scale image.
        
    else:                                                                         # Load whole image folders
        shape0, shape1,image_channel_count = get_image_shape(args.dataset)
        if str2bool(args.toRGBOpt) :                                              # optional
            image_channel_count = 3

        dataset_path = args.dataset

    '''Setting Configurations''' 
    assert args.command in ['train','retrain' ,'detect',"val"]
    print ("os.path.isdir(args.dataset)",os.path.isdir(args.dataset))
    if args.command in [ "train" ,"retrain"] :
        config = NucleusConfig(args.dataset)
         # prepare validation ids
        if os.path.isdir(args.dataset): # input is folder
            # import pdb;pdb.set_trace()
            fns_datasets = os.listdir(os.path.join(args.dataset))    
            random.shuffle(fns_datasets)
            VAL_IMAGE_IDS  = fns_datasets[0:int(len(fns_datasets)*0.05)]
            config.VAL_IMAGE_IDS = VAL_IMAGE_IDS
        #else:       ## input is a file

        config.set_IMAGE_CHANNEL_COUNT(image_channel_count )
        config.set_IMAGES_PER_GPU(5)                                               # in case memory error, default is 15

        if args.fgmask is not None:                                                # cast the mask on image ,only detect the foreground part of the image
            if ".npy" in args.fgmask:
                fg_mask = np.load( args.fgmask )
            elif ".txt" in args.fgmask:            
                fg_mask = np.loadtxt( args.fgmask ,delimiter=",", dtype = int)
            elif ".h5" in args.fgmask: 
                hf = h5py.File(args.fgmask, 'r')                                                             # load wholelabel use 9s
                fg_mask = np.array(hf.get('seg_results'))
                hf.close()
            fg_mask = np.dstack( [fg_mask >0]*3)
            wholeImage = wholeImage*fg_mask
            print ("args.fgmask =",args.fgmask )
            del fg_mask

    else:  # detectubg
        if args.command == "detect":
            image_detection_full_name = os.path.join( args.results , "SegInput.tif")        
            tiff.imsave (image_detection_full_name,wholeImage)
            dataset_path = image_detection_full_name
            del wholeImage,rawImage

        config = NucleusInferenceConfig(args.dataset)
        config.set_IMAGE_CHANNEL_COUNT(image_channel_count )        
        config.set_IMAGES_PER_GPU(int(args.batch_size) )  # in case memory error, default is 15
    
    if os.path.isdir(args.dataset) :  # Load images path
        config.IMAGE_MIN_DIM = shape0
        config.IMAGE_MAX_DIM = shape0
        config.set_crop_size([shape0,shape1])
        config.IMAGE_SHAPE  = (shape0, shape1,image_channel_count )    
    config.display()

    ''' Create model'''
    if args.command == "train":
        print("Logs: ", args.logs)
        model = modellib.MaskRCNN(mode="training", 
                                  config=config,
                                  model_dir=args.logs)
    elif args.command == "retrain":
        print("Logs: ", args.logs)
        model = modellib.MaskRCNN(mode="training", 
                                  config=config,
                                  model_dir=args.logs, retrain=True)
    else:
        model = modellib.MaskRCNN(mode="inference", 
                                  config=config,
                                  model_dir=DEFAULT_LOGS_DIR)
    t_m = time.time() 
    print ("*"*20 + "\n....Model built using time: ", str(t_m- tic))

    # Select weights file to load, if "None"=> train from scratch
    if args.weights is not None: 
        # Load path of weights
        if args.weights.lower() == "coco":
            weights_path = COCO_WEIGHTS_PATH
            # Download weights file
            if not os.path.exists(weights_path):
                utils.download_trained_weights(weights_path)
            model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",  
                "mrcnn_bbox", "mrcnn_mask"])
        elif args.weights.lower() == "last":
            # Find last trained weights
            weights_path = model.find_last()
        elif args.weights.lower() == "imagenet":
            # Start from ImageNet trained weights
            weights_path = model.get_imagenet_weights()            
        else:
            weights_path = os.path.normpath(args.weights)
            if args.command =="retrain":
                weights_path =   load_latest_weight(weights_path)    #             # load the latest weight from the traingLog(weights) folder
        print("------load weights_path=",weights_path)
        model.load_weights(weights_path, by_name=True)


    # assert args.channelOpt in [ "DPH", "RDBH", "D", "H" ]   
    # Train or evaluate

    if args.command in [ "train" ,"retrain"] :
        # Get mask directory from image path
        if os.path.isdir(args.dataset) is False : 
            print ("#################   Dataset = whole image")
            crop_size = [512,512]   # hard code
            global wholelabel

            if ".npy" in args.label:
                wholelabel = np.load( args.label )
            elif ".h5" in args.label: 
                hf = h5py.File(args.label, 'r')                                                             # load wholelabel use 9s
                wholelabel = np.array(hf.get('seg_results'))
                hf.close()
            else:            # txt or csv
                wholelabel = np.loadtxt( args.label ,delimiter=",", dtype = int)
 
            print ("loaded whole brain labels:", args.label, " dtype=", wholelabel.dtype)

            toc1 =  time.time()
            train_model(model, args.dataset, crop_size =crop_size,
                            augmentation= True,toRGB= str2bool(args.toRGBOpt),epoch = int(args.epoch))
            toc2 =  time.time()
            print ("total time = ", int(toc2 - tic))
        else:
            print ("#################    Dataset = Image directory")
            toc1 =  time.time()
            train_model(model, args.dataset,
                       augmentation= True,  masks_fName = args.masks_fName,
                       keyImgID = args.keyImgID, toRGB= str2bool(args.toRGBOpt) ,epoch = int(args.epoch))

            toc2 =  time.time()
            print ("total time = ", int(toc2 - tic))
            

    elif args.command == "detect":
        # Detection 

        if args.results is None:
            RESULTS_DIR = os.path.join(ROOT_DIR, "results/nucleus/")
        else:
            RESULTS_DIR = args.results        
        crop_size = [512,512]

        t_start = time.time()
        print ("Load data used time:", t_start - tic)            

        submit_dir = detect_model(model, dataset_path, crop_size ,
                                    batch_size      = config.BATCH_SIZE,
                                    result_dir      = RESULTS_DIR,
                                    imadjust        = bool(args.imadjust),
                                    toRGB           = str2bool(args.toRGBOpt), 
                                    bgBoost         = int(args.bgBoost),
                                    verbose         = 0
                                    )
        # submit_dir = "/brazos/roysam/xli63/exps/Data/50_plex/jj_final/seg_results_8plex_pretained/submit_20190306T165045"
        del model

        print ("@@@@@@@@@@@@ whole brain detection done!")
        toc1 =  time.time()
        print ("Whole Brain detection used time = ", int(toc1 - tic))
        
        if os.path.isdir(args.dataset) is False :   # load large scale image
            # merge detection results
            print ("************* Merge detection results**********************")
            paras_crop = merge_crop_fcts.Paras()
            paras_crop.set_crop_size (crop_size)
            paras_crop.set_img_shape( [wholeImage_shape[0],wholeImage_shape[1]])
            paras_crop.display()

            if args.visimg is None:
                with tiff.TiffFile(dataset_path) as tif:
                    display_img  = tif.asarray(memmap=True)
                if display_img.ndim == 2:
                    display_img = skimage.color.gray2rgb(display_img)
            elif os.path.isdir(args.visimg):    # the visimage is a folder
                image_dict = {}
                for i, image_path in enumerate( os.listdir(args.visimg) ) :
                    if ".tif" in image_path:
                        with tiff.TiffFile(os.path.join(args.visimg , image_path)) as tif:
                            image_dict [i]  = tif.asarray(memmap=True)
                display_img = image_dict
            else:                               # the visimage is a file
                with tiff.TiffFile(args.visimg) as tif:
                    display_img  = tif.asarray(memmap=True)
                if str2bool(args.imadjust):                
                    display_img =  dt_utils.image_adjust(display_img)          # Load over images

            merged_label = merge_crop_fcts.result_merge( whole_image     = display_img , 
                                                        submit_file      = submit_dir ,
                                                        write_folderName = RESULTS_DIR,
                                                        paras            = paras_crop,
                                                        verbose          = 0)            
            toc2 =  time.time()
            print ("Merge detection used time = ", int(toc2 - toc1))

            if args.val_crops is not None:
                pred_path = os.path.join( RESULTS_DIR , "pred_atlas") 
                os.makedirs( pred_path,exist_ok= True)
                if ".csv"  in args.val_crops:
                    df_names = pd.read_csv(args.val_crops)
                    save_fig = df_names["crop_img"]. tolist()
                    dt_utils.annotation(merged_label, 
                                        display_img[:,:,2],   # DPH gray scale image as background for annotation
                                        save_fig,
                                        pred_path)

                os.makedirs( os.path.join(RESULTS_DIR,"eval"),exist_ok= True)
                # dt_utils.iou_ann(pd_path=pred_path,
                #                 gt_path = "/project/hnguyen/xiaoyang/exps/Data/50_plex/jj_final/atlas/ground_truth",
                #                 submit_df = os.path.join(RESULTS_DIR,"merged_submission.csv"),
                #                 output_dir= os.path.join(RESULTS_DIR,"eval"),
                #                 )
            os.remove(image_detection_full_name)   # remove the SegInput.tif

    elif args.command ==  "val":
        assert os.path.isdir(args.dataset)                                                                # save segmentation results instantly 
        
        savefig_dir = os.path.join(  args.results,  "pred_atlas")
        os.makedirs(savefig_dir,exist_ok=True)

        submit_dir = detect_model(model,  args.dataset ,
                                    batch_size      = config.BATCH_SIZE,
                                    result_dir      = args.results,
                                    imadjust        = bool(args.imadjust),
                                    toRGB           = str2bool(args.toRGBOpt), 
                                    bgBoost         = int(args.bgBoost),
                                    savefig_dir     = savefig_dir,
                                    verbose         = 1

                                    )
        # dt_utils.iou_ann(pd_path= savefig_dir,
        #                 gt_path = "/project/hnguyen/xiaoyang/exps/Data/50_plex/jj_final/atlas/ground_truth",
        #                 output_dir= os.path.join(args.results,"eval"),
        #                 submit_df=submit_dir
        #                 )

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))

    print ("\n\nWhole running used time = ", int(time.time() - tic))


