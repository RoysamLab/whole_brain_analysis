# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 13:48:41 2018

Load segmentation result of HPC whole crop  located at "F:\FACS-SCAN_rebeccaNIH2017Summer\Dragan50CHN\SingleSectionsFor2DImageAnalysis\HPC_whole"

Crop the images, create the local xml files in cropped/xml,cropped/img 

1) Create BBox.xml
2) load into interface 
3) correct bboxes



@author: Xiaoyang.Rebecca.Li@gmail.com
"""

import os
import numpy as np
from skimage import exposure,filters, morphology,measure,feature,io,color
from math import sqrt
import sys
import cv2

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import tostring, Element, SubElement
from xml.dom import minidom

os.chdir('F:\FACS-SCAN_rebeccaNIH2017Summer\RebeccaCode')
sys.path.insert(0, (os.getcwd())  +'./' + 'lib_fcts')
import Cropfcts as Cropfcts
import visual_fcts as Visual_fcts

#%% Read raw segmentation result from label text file 
Read_img_file_Loc = 'F:\FACS-SCAN_rebeccaNIH2017Summer\Dragan50CHN\SingleSectionsFor2DImageAnalysis\HPC_whole'

labels = np.loadtxt (Read_img_file_Loc +'\\'+'[DAPI+Histones]soma_Mask.out',delimiter  = ',' , dtype = int)
PropertyTable = measure.regionprops(labels) 

#%% Crop the images into 300*300
Write_Crops_file_loc = Read_img_file_Loc + './' + 'crops' 
if os.path.isdir (Write_Crops_file_loc) ==False :
    os.makedirs(Write_Crops_file_loc)       

def cropWindowXY(cropRange,windowSize):
    #cropRange =  [[ymin, ymax],[xmin,xmax]]  
    cropRange_Ls = []
    for i in range(0, int( (cropRange[0][1] - cropRange[0][0] )/ windowSize[0]) )  :
        for j in range(0, int( (cropRange[1][1] - cropRange[1][0] )/ windowSize[1]) ):
            cropRange_Ls.append ( np.array([ 
                                                [ cropRange[0][0] + windowSize[0] *i , cropRange[0][0] + windowSize[0] * (i + 1) -1],
                                                [ cropRange[1][0] + windowSize[1] *j , cropRange[1][0] + windowSize[1] * (j + 1) -1] 
                                                ] ))  
#    print(cropRange_Ls)
    return (cropRange_Ls)       

OringalImg = cv2.imread(Read_img_file_Loc + './' + '[Histones]_8bit.tif')
imageRange = [ [0,np.shape(labels)[0]],  [0,np.shape(labels)[1]] ]
cropRange_Ls = cropWindowXY(imageRange,windowSize = [300,300]  )
#print(cropRange_Ls[-1:])

for cropRange in cropRange_Ls :
    cropped_labels =  Cropfcts.cropping (labels, cropRange)
    cropped_image =  Cropfcts.cropping (OringalImg, cropRange)
    cropped_image =  Visual_fcts.imadjust(cropped_image[:,:,1])
#    cropped_Name = str(cropRange[0]) + str(cropRange[1])
    cropped_Name = str(cropRange[0][0]) + '_' +str(cropRange[1][0])
    np.save     ( Write_Crops_file_loc + './' +  'npy_dir' + './' + cropped_Name + '.npy' , cropped_labels)
    cv2.imwrite ( Write_Crops_file_loc + './' +  'img_dir' + './' + cropped_Name + '.jpg' , cropped_image)   


#%% Write Bbox into dir

def prettify(elem):
    """Return a pretty-printed XML string for the Element. """
    rough_string = tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def writeBBoxXML (img_dir,npy_dir):    
    ''' create a new folder parella to imgdir storage the BBox XMLs'''
    bbox_dir = os.path.dirname(img_dir) + './' + 'bbox_dir' 
    if os.path.isdir (bbox_dir) ==False :
        os.makedirs(bbox_dir)  
        
#    npyFileName  ='[ 9900 10199][  0 299].npy'   
    for npyFileName in os.listdir(npy_dir):
#        print(npyFileName)
        # load labels
        labels = np.load (npy_dir + './' + npyFileName)  
        
        if labels.max() > 0:            
            PropertyTable = measure.regionprops(labels) 
            # load images
            imageName = npyFileName.split('.')[0] + '.jpg'
            image = cv2.imread (img_dir + './' + imageName)
            # load X corner, Y corner            
#            Y_cor  = int (npyFileName.split(']')[0].split('[')[1].split(' ')[-2] ) 
#            X_cor  = int (npyFileName.split('[')[2].split('.')[0].split(' ')[-2] ) 
#            
            # Write XML
            rootET = Element('annotation')    
            folder = SubElement(rootET, 'folder')
            folder.text = 'img_dir'
            filename = SubElement(rootET, 'filename')
            filename.text = imageName
            path = SubElement(rootET, 'path')
            path.text = img_dir
            source = SubElement(rootET, 'source')
            database = SubElement(source, 'database')
            database.text = 'Unknown'    
            size = SubElement(rootET, 'size')
            width = SubElement(size, 'width')
            width.text = str( image.shape[0] ) 
            height = SubElement(size, 'height')
            height.text = str( image.shape[1] ) 
            depth = SubElement(size, 'depth')
            depth.text = '1'    
            segmented = SubElement(rootET, 'segmented')
            segmented.text = '0'
        
        
            for obj in PropertyTable:
                Object = SubElement(rootET, 'object')
                name = SubElement(Object, 'name')
                name        = SubElement(Object, 'name')    
                pose        = SubElement(Object, 'pose')
                truncated   = SubElement(Object, 'truncated')
                difficult   = SubElement(Object, 'truncated')
                pose.text       = 'Unspecified'
                truncated.text  = '0'
                difficult.text  = '0'        
                bndbox      = SubElement(Object, 'bndbox')
                xmin        = SubElement(bndbox, 'xmin')
                ymin        = SubElement(bndbox, 'ymin')
                xmax        = SubElement(bndbox, 'xmax')
                ymax        = SubElement(bndbox, 'ymax')        
                name.text      = str(obj.label)   
                ymin.text      = str(obj.bbox[0] ) 
                xmin.text      = str(obj.bbox[1] )
                ymax.text      = str(obj.bbox[2] )
                xmax.text      = str(obj.bbox[3] )       
                
            #print (prettify(rootET))    # display in screen            
            xml_fileName = filename.text.split('.')[0] + '.xml'     
            xml_file = open(bbox_dir+'./' + xml_fileName,'w')    
            xml_file.write(prettify(rootET))
            xml_file.close()

img_dir =  r'F:\FACS-SCAN_rebeccaNIH2017Summer\Dragan50CHN\SingleSectionsFor2DImageAnalysis\HPC_whole\crops\img_dir'
npy_dir =  r'F:\FACS-SCAN_rebeccaNIH2017Summer\Dragan50CHN\SingleSectionsFor2DImageAnalysis\HPC_whole\crops\npy_dir'
writeBBoxXML(img_dir,npy_dir)

