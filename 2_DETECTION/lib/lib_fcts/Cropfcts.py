# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:59:38 2017

@author: xli63
"""

import os

#os.chdir(r'D:\research in lab\NIHIntern(new)\RebeccaCode')  # set current working directory
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import numpy as np
import skimage
from skimage import util,segmentation,exposure,filters, morphology,measure,feature,io
from scipy import ndimage,stats,cluster,misc,spatial
from sklearn.cluster import KMeans
from sklearn.neighbors  import NearestNeighbors

import numpy as np
import cv2
import heapq
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from math import sqrt

import sys

#sys.path.insert(0, os.getcwd()+'/lib_fcts')
#import CellSegmentationfcts as myfcts                                          # my functions
#from Dataset_read_TBI_BFcorrected import Dataset_read_TBI                                  # read file proterty files
#ds = Dataset_read_TBI()                                                            # define a class to read the input channels image(image name and paramaters)
def checkPath(loc):
	print ('aaaaaaaa')
	if os.path.isdir(loc) == False:
		os.makedirs(loc)    
		
	if sys.platform == "linux" or sys.platform == "linux2":
		if '\\' in loc :
			locLinux = loc.replace('\\', '/')
			return locLinux
	else:
		return loc
	

def imadjust(img, tol=[0.01, 0.99]):
    # img : input one-layer image (numpy array)
    # tol : tolerance, from 0 to 1.

    assert len(img.shape) == 2, 'Input image should be 2-dims'

    if img.dtype == 'uint8':
        nbins = 255
    elif img.dtype == 'uint16':
        nbins = 65535

    N = np.histogram(img, bins=nbins, range=[0, nbins])      # get histogram of image
    cdf = np.cumsum(N[0]) / np.sum(N[0])                     # calculate cdf of image
    ilow = np.argmax(cdf > tol[0]) / nbins                   # get lowest value of cdf (normalized)
    ihigh = np.argmax(cdf >= tol[1]) / nbins                 # get heights value of cdf (normalized)

    lut = np.linspace(0, 1, num=nbins)                       # create convert map of values
    lut[lut <= ilow] = ilow                                  # make sure they are larger than lowest value
    lut[lut >= ihigh] = ihigh                                # make sure they are smaller than largest value
    lut = (lut - ilow) / (ihigh - ilow)                      # normalize between 0 and 1
    lut = np.round(lut * nbins).astype(img.dtype)            # convert to the original image's type

    img_out = np.array([[lut[i] for i in row] for row in img])  # convert input image values based on conversion list

    return img_out
#        loc.replace(r"\\",r "/")

def zeropadding(img, canvas_size = (100,100) ) :
    # canvas size must larger than img size
    edgeWidth = int ( (canvas_size[0] - img.shape[0] ) / 2 )  
    edgheight = int ( (canvas_size[1] - img.shape[1] ) / 2 )  
    
    if len(img.shape) ==3 :
        paddedImg = np.zeros( (canvas_size[0],canvas_size[1],3))
        
    else :
        paddedImg = np.zeros((canvas_size[0],canvas_size[1]))


#    if edgeWidth < 0 or edgheight > 0:
    paddedImg[edgeWidth : edgeWidth + img.shape[0],
            edgheight : edgheight + img.shape[1]] = img

              
    return  paddedImg


def cleanEdgedComponents(label, edgeWidth = 1, remainConnectedBd = True):
#'''
#Input: label (pixel label array) e.g label = np.array ( [0,0,0,0,1],
#                                      0,2,0,0,0;
#                                      0,2,2,0,3;
#                                      0,0,0,3,3] )
#        edgeWidth =1
#Output: clean the componnets touching the edge 
#       e.g                           [0,0,0,0,0;
#                                      0,2,0,0,0;
#                                      0,2,2,0,0;
#                                      0,0,0,0,0]       
#       
#''' 
    cleaned_label = np.copy(label)
    # create a edgeFrame array, edge pixels = -1, others =0
    edgeFrame = np.zeros_like (label)-1    
    temp_content = np.zeros((np.shape(label)[0] -2* edgeWidth,   np.shape(label)[1] -2* edgeWidth ) ) 
    edgeFrame[edgeWidth : edgeWidth + temp_content.shape[0],
                edgeWidth : edgeWidth + temp_content.shape[1]] = temp_content
    
#    plt.figure(),plt.imshow(edgeFrame)
    
    edged_labelpixels = label * edgeFrame
    edged_label_ids = -np.unique(edged_labelpixels)      # contain 0 
    
    maxarea = 0
    maxarea_ID =0
    for obj in measure.regionprops(label):
        if obj.area> maxarea:
            maxarea_ID =  obj.label       # storage the max area ID , incase all label pixel has been deleted
            maxarea = obj.area
            
        if obj.label in edged_label_ids :   
            for i in  range (obj.coords.shape[0]):                # for each pixel in the crops
                cleaned_label[ obj.coords[i,0],
                              obj.coords[i,1] ]  = 0   # make all the pixel to 0  backgroud   
    if maxarea_ID !=0: 
        if cleaned_label.sum() == 0 : # storage the max area ID , incase all label pixel has been deleted
            cleaned_label = (cleaned_label + 1) * ((label==maxarea_ID)*1) *maxarea_ID      # make the label of center connected component == largest label of orignial components
    
    if remainConnectedBd == False:     
        cleaned_label =  (cleaned_label > 0) * cleaned_label.max()    # make it the same to the center ID
    else:
        cleaned_label = label *(cleaned_label > 0)
    
    return cleaned_label
                              
def cropping (original_image, cropRange):
    if len(cropRange) == 4 :  # in the form of    (min_row, min_col, max_row, max_col), need to change  to  [[ymin, ymax],[xmin,xmax]]    
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
           
def cropImg_12N (originalImgName,Write_Cropimg_file_Loc ,  cropRange_ls ,crop_Names):   # big crop in cortex region 
#    for originalImgName  in os.listdir(Read_img_file_Loc) :   # 
#        if 'tif' in originalImgName and ("registered" in originalImgName)==False :
#        if ('tif' in originalImgName) and ('BG' in originalImgName) == False and ('meshGrid' in originalImgName) == False :
#        if ('bordered' in originalImgName) or  ('gray' in originalImgName)  : 
#            print (originalImgName)
#        if '8bit.tif' in originalImgName :
#            Write_Cropimg_file_Loc = Write_Cropimg_file_Loc_root + '\\' + originalImgName.split('.')[0] 
	if os.path.isdir(Write_Cropimg_file_Loc) == False:
		os.makedirs(Write_Cropimg_file_Loc)       

	original_image = io.imread(originalImgName)
	
	for cropRange, crop_Name in zip(cropRange_ls,crop_Names):
		image = cropping (original_image, cropRange)

		imagefileName = Write_Cropimg_file_Loc+'\\'+ crop_Name 
#            if ('bordered' in originalImgName ) or ( 'gray' in originalImgName):
		writeImgdone = cv2.imwrite(imagefileName + '.jpg' , image)      # save .jpg only for the direct input for bounding box detection
#                writeImgdone = cv2.imwrite(imagefileName + '.tif' , image)  
		
		if writeImgdone != True:
#			print('Generate Cropped Images for '+ originalImgName +' done!')
#		else:
			print('[Caucious! ]  Generate Cropped Images  failed!!!!!!!!!!!!!!')

def cropNpy_12N (originalNpyName,Write_Cropimg_file_Loc_root ,  cropRange_ls ,crop_Names):   # big crop in cortex region 				
				        
#	Write_Cropimg_file_Loc = Write_Cropimg_file_Loc_root + '\\' + originalNpyName.split('.')[0] 
		
	labelName = originalNpyName
	origninal_label = np.load(labelName)
	
	for cropRange, crop_Name in zip(cropRange_ls,crop_Names):
		# Write npy
		Write_Cropimg_file_Loc = Write_Cropimg_file_Loc_root + '\\' +  'npy'    
		if os.path.isdir(Write_Cropimg_file_Loc) == False:
			os.makedirs(Write_Cropimg_file_Loc)       
		label = origninal_label[cropRange[0][0]:cropRange[0][1],cropRange[1][0]:cropRange[1][1]]     
		NumOfCell = len(np.unique(label))-1
		labelfileName = Write_Cropimg_file_Loc + '\\'+ crop_Name  + '.npy'
		np.save( labelfileName, label)  
		
		# Write cleared label npy
		Write_Cropimg_file_Loc = Write_Cropimg_file_Loc_root + '\\' +  'cleanEdge_npy'    
		if os.path.isdir(Write_Cropimg_file_Loc) == False:
			os.makedirs(Write_Cropimg_file_Loc)       
		cleaned_label = cleanEdgedComponents(label, edgeWidth = 1)
		labelfileName = Write_Cropimg_file_Loc + '\\'+ crop_Name  + 'cleanEdge.npy'
		np.save( labelfileName, cleaned_label)  

		   
		   
		   
def visualize(img, labels =[]):  
# img: grayscale images 
# labels : 
#    labeled image: 0: backgourd, 
#                   1,2,3... : components ID  
# seeds_markered images = seeds for disk(3) 
#filledCell,seeds_marker,labels

   ## visualize the result of blobs to adjust parameters     
    img_adj =  skimage.img_as_float(img)
    img_adj =  exposure.rescale_intensity(img_adj)
    img_disp = cv2.merge((img_adj,img_adj,img_adj))
    

    border = segmentation.find_boundaries(labels)
    img_disp = img_disp.copy()
    img_disp[border==1]=[0,1,0]                                             # watershed mask border :green
#    plt.figure(), plt.imshow(img_disp_seedsAndBorder)

    return img_disp


##''' input image : DAPI + HISTONE (pixel summation) '''   
def imSummation (img1,img2, outputformat = '16bit') :  # save size  
    imSum = np.add(img1, img2)
    imSum = imSum/ imSum.max()                  # float    
    
    if outputformat ==  '16bit':
        imSum = skimage.img_as_uint(imSum)   # change to 16bit 
        
    return imSum





def write_object(f, xmin, ymin, xmax, ymax):

	line = '\t<object>\n'
	f.writelines(line)

	line = '\t\t<name>seed</name>\n'
	f.writelines(line)

	line = '\t\t<pose>Unspecified</pose>\n'
	f.writelines(line)

	line = '\t\t<truncated>0</truncated>\n'
	f.writelines(line)

	line = '\t\t<difficult>0</difficult>\n'
	f.writelines(line)

	line = '\t\t<bndbox>\n'
	f.writelines(line)

	line = '\t\t\t<xmin>' + xmin + '</xmin>\n'
	f.writelines(line)

	line = '\t\t\t<ymin>' + ymin + '</ymin>\n'
	f.writelines(line)

	line = '\t\t\t<xmax>' + xmax + '</xmax>\n'
	f.writelines(line)

	line = '\t\t\t<ymax>' + ymax + '</ymax>\n'
	f.writelines(line)


	line = '\t\t</bndbox>\n'
	f.writelines(line)

	line = '\t</object>\n'
	f.writelines(line)

def write_xml(folderRoot,xml_fname, txt_fname):
	f = open( folderRoot + './xml/' + xml_fname,'w')

	line = '<annotation>\n'
	f.writelines(line)

	line = '\t<folder>gray</folder>\n'
	f.writelines(line)

	line = '\t<filename>' + txt_fname.split('.')[0] + '.jpg' + '</filename>\n'
	f.writelines(line)

#	path = 'D:\\Hengyang\\Weekly_Report\\Week9_10292017\\Segmentation_improvement\\better-seed\\DeepSeedsDetection\\Train\\gray\\'
	grayPath = folderRoot + '\\gray\\'
	line = '\t<path>' + grayPath + txt_fname.split('.')[0] + '.jpg' + '</path>\n'
	f.writelines(line)

	line = '\t<source>\n'
	f.writelines(line)

	line = '\t\t<database>Unknown</database>\n'
	f.writelines(line)

	line = '\t</source>\n'
	f.writelines(line)

	line = '\t<size>\n'
	f.writelines(line)

	line = '\t\t<width>250</width>\n'
	f.writelines(line)

	line = '\t\t<height>200</height>\n'
	f.writelines(line)

	line = '\t\t<depth>3</depth>\n'
	f.writelines(line)

	line = '\t</size>\n'
	f.writelines(line)

	line = '\t<segmented>0</segmented>\n'
	f.writelines(line)

	# write the object information
	f_txt = open(folderRoot + './txt/' + txt_fname, 'r')
	lines = f_txt.readlines()
	f_txt.close()

	for line in lines:
		[ymin, xmin, ymax, xmax] = line.rstrip().split(',')
		write_object(f, xmin, ymin, xmax, ymax)

	line = '</annotation>\n'
	f.writelines(line)

	f.close()



