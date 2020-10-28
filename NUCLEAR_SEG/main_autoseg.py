'''
Automatic Segmentation on Large scale gray scale image
Author: Rebecca LI, University of Houston, Farsight Lab, 2019
xiaoyang.rebecca.li@gmail.com

e.g.
1) Run  autoseg with LoG Seeds

    cd /brazos/roysam/xli63/exps/SegmentationPipeline/Automatic_Seg

    #pip install fcswrite --user
    data_dir="/brazos/roysam/50_plex/Set#1_S1"

    python main_autoSeg_script.py \
    --dataset="$data_dir"/final \
    --results=/brazos/roysam/xli63/exps/Data/50_plex/jj_final/autoseg


2) Run autoseg with customized seeds

    cd /brazos/roysam/xli63/exps/SegmentationPipeline/Automatic_Seg

    data_dir="/brazos/roysam/50_plex/Set#1_S1"

    python main_autoSeg_script.py \
    --dataset="$data_dir"/final \
    --results=/brazos/roysam/xli63/exps/Data/50_plex/jj_final/autoseg_seedsProvides

'''

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import numpy as np
import skimage
from skimage import util,segmentation,exposure,filters, morphology,measure,feature,io
from scipy import ndimage,stats,cluster,misc,spatial

import xml.etree.ElementTree as ET

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from math import sqrt

import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'autoseg'))
import CellSegmentationfcts_hierarchy_wholebrain as myfcts                                          # my functions
import GenerateICE_updated50CHN as ice  
import time
import warnings
import tifffile as tiff
from memory_profiler import profile

def warningIgnore():
    warnings.filterwarnings("ignore")

def imadjust16bit(img):
    img = img/ (img.max())    # float
    img = skimage.img_as_uint(img)    # 16 bit
    return img

##''' input image : DAPI + HISTONE (pixel summation) '''   
def imSummation (img1,img2) :  # save size  
    imSum = np.array(np.add(img1, img2),dtype = float)
    imSum = imSum/ imSum.max()            # float
    imSum = skimage.img_as_uint(imSum)   # change to 16bit 
    
    return imSum

def convertCirclelabel(label , blob , seeds_marker):
    imgLabel = np.zeros_like(label)
    
    x, y = np.indices(label.shape)
    for i in  range (blob.shape[0]):    
        x_i = blob[i,0]
        y_i = blob[i,1]
        r_i =(blob[i,2]) 
        r_i_adjusted = r_i * 3.6/np.log(r_i) 
        rr, cc = skimage.draw.circle( x_i,y_i,r_i_adjusted)
        r_bdId =  rr< label.shape[0]                             # fixin the boundary of image
        c_bdId =  cc< label.shape[1] 
        bdId = np.logical_and(r_bdId,c_bdId)   

        imgLabel[rr[bdId] , cc[bdId] ] = 1
        
    D = ndimage.distance_transform_edt(imgLabel)    
    
    circularLabel = morphology.watershed(-D, seeds_marker,  mask = imgLabel)       
    

    
    return circularLabel

def convertEllipsoidlabel (label, blob, seeds_marker):
    imgLabel = np.zeros_like(label)
    
    x, y = np.indices(label.shape)
    for obj in   measure.regionprops(label):    
        r,c = obj.centroid
        r_radius = obj.minor_axis_length    /2
        c_radius = obj.major_axis_length    /2
        r_radius_adjusted = r_radius * 3.5/np.log(r_radius) 
        c_radius_adjusted = c_radius * 3.5/np.log(c_radius) 
        ori = obj.orientation 
        rr, cc = skimage.draw.ellipse(int(r), int(c), int(r_radius_adjusted), int( c_radius_adjusted), rotation= ori)
        r_bdId =  rr< label.shape[0]              # fixin the boundary of image
        c_bdId =  cc< label.shape[1] 
        bdId = np.logical_and(r_bdId,c_bdId)   

        imgLabel[rr[bdId] , cc[bdId] ] = 1
        
    D = ndimage.distance_transform_edt(imgLabel)    
    
    ellipsoidLabel = morphology.watershed(-D, seeds_marker,  mask = imgLabel)       
    

    
    return ellipsoidLabel

def extractFileNamesforType(dir_loc, fileExt):   # dir , '.tif'
    readNames =    os.listdir (dir_loc)
    types = np.array( [ ([ x.find(fileExt)>0 for x in  readNames ]) ] , dtype = int)   # only extract extension .tif
    typesIDs = np.array  (  np.where(types > 0)[1] )
    fileNames = []
    ([fileNames.append(readNames[i]) for i in typesIDs])    
    return fileNames
           

def visualize(img, seeds_marker = [],labels =[],seeds_marker_2 = []):  
   
    img_adj =  skimage.img_as_float(img)
    img_adj =  exposure.rescale_intensity(img_adj)
    img_disp = cv2.merge((img_adj,img_adj,img_adj))
    
    img_disp_Onlyseeds = img_disp.copy()
    img_disp_OnlyBorders = img_disp.copy()

    if labels !=[]:
        border = segmentation.find_boundaries(labels)
        img_disp[border==1]=[0,1,0]                                             # watershed mask border :green
        img_disp_OnlyBorders[border==1]=[1,0.5,0.8]                                             # watershed mask border :pink
    
    if seeds_marker !=[]:        
        img_disp[seeds_marker > 0]=[1,0,0]                                      # detected seeds        :green
        img_disp_Onlyseeds[seeds_marker > 0]=[1,0,0]       
        
        if seeds_marker_2 != []:
            img_disp[seeds_marker_2 > 0]=[0,0,1]                                      # detected seeds        :red
         
    return img_disp, img_disp_Onlyseeds,img_disp_OnlyBorders


def writeTifandBin(Write_img_file_Loc,fileNm, IMG, label, seeds_marker =[], display = False, saveDisp =False, writeBin = True): 

    labelLocFileName = []
    
    if display == True:                # only display,no write
            #img_disp, img_disp_Onlyseeds,img_disp_OnlyBorders = visualize(img, seeds_marker = [],labels =[])
        img_disp,__,__ = visualize(IMG , seeds_marker=[] ,labels = label)    
        rgb = skimage.color.label2rgb(label,image=None, bg_label=0, bg_color=(0, 0, 0))

    if saveDisp ==True :  
        img_disp,img_disp_seeds,__  = visualize(IMG , seeds_marker =[] ,labels =label)    
        rgb = skimage.color.label2rgb(label,image=None, bg_label=0, bg_color=(0, 0, 0))
        
        tiffileNm  = (fileNm + '_Mask_DisplayBorder')     
        val_1 = cv2.imwrite (os.path.join(Write_img_file_Loc ,  tiffileNm +'.tif' ),skimage.img_as_ubyte(img_disp))     # display for 8 bit        
                         
        tiffileNm  = (fileNm + '_Mask_DisplayLabel')      
        val_2 = cv2.imwrite (os.path.join(Write_img_file_Loc ,  tiffileNm +'.tif') ,skimage.img_as_ubyte(rgb))          # display for 8 bit
        
        if seeds_marker !=[]:   
            tiffileNm  = (fileNm + '_Mask_DisplayOnlySeeds')      
            val_3 = cv2.imwrite (os.path.join(Write_img_file_Loc ,  tiffileNm +'.tif') ,skimage.img_as_ubyte(img_disp_seeds))          # display for 8 bit
        
        if val_1  ==False or val_2  ==False:
            print('[Warning!!] Write unsuccussfull =',tiffileNm)       
            
    if writeBin == True :
        # only save the colored mask and label.out for none- neuron#            
        # write labelleds image masks
        maskfileNm = (fileNm + '_Mask')           
        labelLocFileName = os.path.join ( Write_img_file_Loc ,  maskfileNm +'.out' )
        np.savetxt(labelLocFileName, label, fmt='%d',delimiter=',')   # X is an array        
        #
    #            write labelled image masks into bins
        labelbinLocFileName = os.path.join ( Write_img_file_Loc ,  maskfileNm + '.bin' )
        fout = open(labelbinLocFileName,'wb')                           
        labels_uint32 = np.uint32(label)                                                                     # change it to uint 32 bit with 4 bytes             
        labels_uint32.tofile(fout)                                                                               # write into .bin file
        fout.close()
       
        
        # check the size of Masks bin files
        maskBinSize = os.path.getsize(labelbinLocFileName)       
        expectedsizeofbin = label.shape[0]* label.shape[1]*32/8        #               # 4byte =32 bit       
    
        print('Numberof objects = ',label.max(),' size of Mask:',maskBinSize,'  locate at: ',labelbinLocFileName)            
    #        print('Actual size of Mask bin file(bytes)  = ',maskBinSize)         
    #        print('Expected size of Mask bin file(bytes)  = ',expectedsizeofbin)         
    
        if maskBinSize != expectedsizeofbin:                      # the size of the binary file shall equal to ( #of objects * bitdepth) /8
            print ('[Error !] FeatureTable into binary image wrong')

    return labelLocFileName
        
############################################################
#  Command Line
############################################################
def str2bool(str_input):
    bool_result = True if str_input.lower() in ["t",'true','1',"yes",'y'] else False
    return bool_result

if __name__ == '__main__':
    import argparse,time
    warningIgnore()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Automatic Segmentation for DAPI, Histone or DAPI+Histone')

    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        default = None,
                        help="Whole image to apply cell segmentation, must contain '*XLSX.xml' in the same folder")
    parser.add_argument('--CHN', required=False,
                        metavar="/path/to/dataset/",
                        default = "DPH",
                        help="Which channel for segmentation 'DAPI','Histone'or 'DPH'")
    parser.add_argument('--seed', required=False,
                        metavar="/path/to/seed/",default= None,
                        help='seed for whole to create borders')                        
    parser.add_argument('--results', required=False,
                        metavar="/path/to/result/",default = None,
                        help='directory for results, default = dataset+"AutoSeg_Results") ')
    parser.add_argument('--demo', required=False,
                        default = '0',type = str, 
                        help='whether to run demo for a small cropped image, not support for provides seeds')
    parser.add_argument('--cast', required=False,
                        default = '0',type = str, 
                        help='whether to cast the nucleus masks on all other channels')

    args = parser.parse_args()

    tic = time.time()


    ''' load channel info configs from .xml  '''
    if args.seed  is not None:
        # os.path.join(path,"bbxs_detection.txt")
        bbxs_detection=np.loadtxt( args.seed,skiprows =1) # ID	centroid_x	centroid_y	xmin	ymin	xmax	ymax
        blobs = bbxs_detection[:,[2,1]]
    else:
        blobs = None

    Read_img_file_Loc = args.dataset
    xmlreadNames = extractFileNamesforType(Read_img_file_Loc, 'XLSX.xml')
    xmlreadName = xmlreadNames[0]      
    tree = ET.parse(os.path.join ( Read_img_file_Loc,  xmlreadName))
    root = tree.getroot()
    Images = root[0]        
    def getAttri(Searchbiomarker,key):    
        output = []
        for Image in  Images.findall('Image'):                            # take the current animal 
            bioMarker = Image.get('biomarker')
            if Searchbiomarker == bioMarker:
                if key == 'FileName':
                    output = Image.find('FileName').text
                elif (key in ['LoG_Paras', 'ImgColor' , 'SeedColor' ] ):  # output the list
                    string =  Image.get(key)
                    string = string.split('[')[1]
                    string = string.split(']')[0]
                    strings = string.split(',')
                    output = np.array(strings,dtype = float )                
                else:
                    output = Image.get(key)
            else:
                pass
                    
        if output ==[]:
            print('Key not founded!')          
            os.system("pause")        
        return output
            
    PropertyTables = {}
    nucleus_blobs_LoG ={}
    nucleus_images = { }
    nucleusBiomarkes = ['DAPI','Histones']    

    if str2bool(args.demo) == True:    # try whole img    .
        cropRange= [[10000,10200],[20200,20600]]      # crop for testing hippocampus  200*400    #        
        CropStr = 'X_'+ str(cropRange[0][0])+ '_' + 'Y_'+ str(cropRange[1][0]) + '_' + 'H_'+ str(cropRange[0][1] - cropRange[0][0]) +'_' + 'W_'+ str(cropRange[1][1]-cropRange[1][0]) 
        if args.results == None:
            Write_img_file_Loc = os.path.join (  Read_img_file_Loc_root , 'AutoSeg_Results-demo-' + CropStr )                       # write in new folder 'Outputs'               
        else:
            Write_img_file_Loc = args.results
        if os.path.exists(Write_img_file_Loc) == False:
            os.makedirs(Write_img_file_Loc)        

        for Image in  Images.findall('Image'):    
            biomarker = Image.get('biomarker')
            CHNName =  Image.find('FileName').attrib['CHNName']
            if CHNName in ['R2C1','R2C2' ]        :        
                imgName = os.path.join ( Read_img_file_Loc ,  Image.find('FileName').text ) 
                original_image = io.imread(imgName)            
                nucleus_images[biomarker] = original_image[cropRange[0][0]:cropRange[0][1],cropRange[1][0]:cropRange[1][1]]                             
                del original_image           # save memory   

    else:    # try whole img    .
        if args.results == None:
            Write_img_file_Loc = os.path.join (Read_img_file_Loc_root ,"AutoSeg_Results"  )                        # write in new folder 'Outputs'                     
        else:
            Write_img_file_Loc = args.results
        if os.path.exists(Write_img_file_Loc) == False:
            os.makedirs(Write_img_file_Loc)    

        for Image in  Images.findall('Image'):    
            #    biomarker = imreadName.split('__')[0]                                                  # biomarker name is inside the fileName    
            biomarker = Image.get('biomarker')
            CHNName =  Image.find('FileName').attrib['CHNName']
            # print ("load config of ", CHNName,  " biomarker = ", biomarker, " FileName=", Image.find('FileName').text )
            if CHNName in ['R2C1','R2C2' ]        :        
                imgName = os.path.join ( Read_img_file_Loc ,  Image.find('FileName').text ) 
                original_image = io.imread(imgName)            
                nucleus_images[biomarker] = original_image                            
                del original_image           # save memory   
        

    if args.CHN == "DPH":            
        DPH = 'DAPI+Histones'
        nucleus_images[DPH] = imSummation(nucleus_images['DAPI'] ,nucleus_images['Histones'] )
        cv2.imwrite(os.path.join(Write_img_file_Loc,'[DAPI+HISTONES]8bit.tif' ),skimage.img_as_ubyte (nucleus_images[DPH]) )


    ''' Run segmentation  '''

    tic = time.clock()
            
    # Compare DAPI,DAPI + Histone and  ImgFillinga
    Write_img_file_Loc =os.path.join ( Write_img_file_Loc ,  args.CHN)
    if os.path.isdir(Write_img_file_Loc) == False:
        os.makedirs(Write_img_file_Loc)
    # for key,binaryPar in zip(  ['DAPI','Histones',DPH],[0.4,0.4,0.6] ):   # compare DAPI and DAPI+HISTONE
    for key,binaryPar in zip(  [args.CHN],[0.6] ):   # compare DAPI and DAPI+HISTONE

        img = nucleus_images[key]
        img = imadjust16bit(img)
        for imagefilling in [False]:            

            if imagefilling == True:
                cell = filters.gaussian(img, sigma=1)
                cell = myfcts.fillinhole((img) ,secondRd = True , binaryPar = binaryPar)   # do image filling outside the function

            else:
                cell = filters.gaussian(img, sigma=1)

            print (" Start to run watershedSegmentation")    
            seeds_marker, labels, blobs_nuclues =  myfcts.watershedSegmentation(img = cell,  blobs = blobs,
                                maskCorrectR = 3, maskDilateR =0 , LoG_Para = getAttri('DAPI','LoG_Paras'), 
                                bgBoost = False, offset = 0.25, imagefilling = False)
            print (" watershedSegmentation finished")    
            print('watershedSegmentation use time(s) :', ( '{:.2f}'.format(time.clock()- tic) ))
            import pdb ; pdb.set_trace()    
            np.save("seeds_marker.npy",seeds_marker)
            np.save("labels.npy",labels)
            np.save("blobs_nuclues.npy",blobs_nuclues)
            
            NumOfCell = blobs_nuclues.shape[0] if type(blobs_nuclues)==np.ndarray else len(blobs_nuclues)
            del nucleus_images, img,blobs_nuclues         # save memory   

            img_disp, img_disp_Onlyseeds,img_disp_OnlyBorders = visualize(cell, seeds_marker = seeds_marker,labels =labels)
            print('visualize use time(s) :', ( '{:.2f}'.format(time.clock()- tic) ))


            filename = '['+ key +']' + str(NumOfCell)+ 'seeds_' + 'fill_'+ str(imagefilling) 
            tiff.imsave(os.path.join ( Write_img_file_Loc ,  filename +'.tif' ), skimage.img_as_ubyte(img_disp), bigtiff=True)

            print (" imwrite finished")    

            # filename = '['+ key +']' + str(NumOfCell)+ 'seeds_' + 'fill_'+ str(imagefilling) + '_onlySeeds'
            # cv2.imwrite(os.path.join ( Write_img_file_Loc ,  filename +'.tif' ), skimage.img_as_ubyte(img_disp_Onlyseeds))

            # filename = '['+ key +']' + str(NumOfCell)+ 'seeds_' + 'fill_'+ str(imagefilling) + '_onlyBorders'
            # cv2.imwrite(os.path.join (Write_img_file_Loc ,  filename +'.tif'), skimage.img_as_ubyte(img_disp_OnlyBorders ))

            filename = '['+ key +']' + str(NumOfCell)+ 'seeds_' + 'fill_'+ str(imagefilling) 
            # np.save(os.path.join (Write_img_file_Loc ,  filename + 'labels_nuclear.npy'), labels)
            np.savetxt (os.path.join(Write_img_file_Loc, filename + 'labels_nuclear.txt'),labels , fmt ='%d',delimiter=',',)

            toc1 = time.clock()
            print('Generate masks of [',key,' ] use time(s) :', ( '{:.2f}'.format(toc1- tic) ))
            print ('Results save in  ', Write_img_file_Loc)
        
    saveDisp = False  
    
    if saveDisp == True:
        labels_dilated = morphology.dilation(labels, morphology.disk(4) )    # the thickness of plasma

        border = segmentation.find_boundaries(labels)
        border = morphology.binary_dilation(border, morphology.disk(1) )             # 1 for border, 0 for background
        labels_border = labels * border                                       # labeld thick border  
        labels_plasma  = labels_dilated*(labels==0) + labels_border 
        labels_soma = labels_plasma*(labels==0) + labels     
        
        np.save(os.path.join (Write_img_file_Loc ,  'labels_nuclear.npy'), labels)
        labels_nuclear_LocFileName     =   writeTifandBin(Write_img_file_Loc, DPH + 'nuclear' , 
                                                        nucleus_images['DAPI+Histones'], labels , seeds_marker, 
                                                        display = False,saveDisp = saveDisp,writeBin = False)
        
        np.save(os.path.join (Write_img_file_Loc ,  'labels_soma.npy'), labels_soma)
        labels_soma_LocFileName     =   writeTifandBin(Write_img_file_Loc, DPH + 'soma' , 
                                                        nucleus_images['DAPI+Histones'], labels_soma , seeds_marker, 
                                                        display = False,saveDisp = saveDisp,writeBin = True)
        
        np.save(os.path.join (Write_img_file_Loc ,  'seeds_marker.npy'), seeds_marker)

    toc2 = time.clock()
    print('Write into file use time(s) :', ( '{:.2f}'.format(toc2- toc1) ))
    print ('Results save in  ', Write_img_file_Loc)


   
    
    if str2bool(args.cast) is True:
        print('............. load labels ...........')

        labels =np.load(os.path.join (Write_img_file_Loc ,  'labels_nuclear.npy'))    # load label .npy            
        labels_soma = np.load(os.path.join (Write_img_file_Loc ,  'labels_soma.npy'))
        seeds_marker = np.load(os.path.join (Write_img_file_Loc ,  'seeds_marker.npy'))
        '''Load images one by one to generate image display tif and bins Mask for each channel'''
    
        for Image in  Images.findall('Image'):    
            biomarker = Image.get('biomarker')
            if biomarker != 'NONE' :       #Implement LoG on whole image  only for DAPI channel  
                imreadName = Image.find('FileName').text
                image = io.imread( os.path.join (Read_img_file_Loc, imreadName) )
                if str2bool(args.demo) == True:
                    image = image[cropRange[0][0]:cropRange[0][1],cropRange[1][0]:cropRange[1][1]]     # left cropps
                
                ChannelName = '[' +biomarker+']'    
                
                imWriteName = ChannelName + '_8bit.tif'
                cv2.imwrite (os.path.join (Write_img_file_Loc ,  imWriteName), skimage.img_as_ubyte(image))                      # display for 8 bits

                writeTifandBin(Write_img_file_Loc, ChannelName + 'soma' , image, labels_soma , seeds_marker=[], display = False, saveDisp = saveDisp,writeBin = True)
                PropertyTables[(biomarker +'_somaMasked') ] = measure.regionprops(labels_soma,intensity_image=image)                  # storage the properties e.g ID,area  of each componentes (labelled regions)
                
                writeTifandBin(Write_img_file_Loc, ChannelName + 'nucleus' , image, labels , seeds_marker=seeds_marker, display = False, saveDisp = saveDisp,writeBin = False)
                PropertyTables[(biomarker +'_nucleusMasked') ] = measure.regionprops(labels,intensity_image=image)                  # storage the properties e.g ID,area  of each componentes (labelled regions)

        
        '''Write property table in to files '''
        textTable_ObjID = {}
        NumOfObj =  labels_soma.max()               # centriod coording to soma mask, should be the fullest 
        PTs_Keylist = list (PropertyTables.keys())
        #generate ID s
        textTable_ObjID['ID'] = np.arange(NumOfObj) +1
        
        textTable_ObjID['centroid_x'] = np.zeros(NumOfObj,dtype = np.uint32)
        textTable_ObjID['centroid_y'] = np.zeros(NumOfObj,dtype = np.uint32)
        
        for obj in PropertyTables[PTs_Keylist[0]] :
            label_ID = obj.label
            textTable_ObjID['ID'][label_ID-1]           = obj.label
            textTable_ObjID['centroid_x'][label_ID-1]   = image.shape[0] - np.uint32 (obj.centroid[0])   # flip over the crop
            textTable_ObjID['centroid_y'][label_ID-1]   = np.uint32 (obj.centroid[1])		
            
        # generate all the other features	
        textTable_Content = {}
        for PropertyTable in PropertyTables.items():
            maskName = PropertyTable[0]    # the mask name
            textTable_Content[maskName + '__Tol_intensity'			] = np.zeros(NumOfObj,dtype = np.float32)
            textTable_Content[maskName + '__Avg_intensity'			] = np.zeros(NumOfObj,dtype = np.float32)

            for obj in  PropertyTable[1]: # the contents  name
                    label_ID = obj.label
                    textTable_Content[maskName + '__Tol_intensity'			][label_ID-1]  = np.float32(obj.intensity_image.sum() )      # Tol_intensity 
                    textTable_Content[maskName + '__Avg_intensity'			][label_ID-1]  = np.float32(obj.mean_intensity)   # mean_intensity 
        
        datasetName = xmlreadName.split('Dat')[0]
        #   write properties of components      into txt.................  for ICE
        featureTableFilename = os.path.join ( Write_img_file_Loc,  datasetName + '_FeatureTable.txt' )
        T = open(featureTableFilename, 'w')        # create a table txt to write   
        T.write('ID' + '\t' + 'centroid_x' + '\t' + 'centroid_y' + '\t' ) #write header
        for Content in textTable_Content.items():
            T.write(Content[0] + '\t' ) 
        T.write('\n')  
        
        #write data
        for i in range(len(textTable_ObjID['ID'])):
            T.write(str( textTable_ObjID['ID'][i]        )    + '\t'                 )      # ID
            T.write(str( textTable_ObjID['centroid_x'][i] )   + '\t'                 )      # centroid_x
            T.write(str( textTable_ObjID['centroid_y'][i] )  + '\t'                 )      # centroid_y    
            for  j, Content in enumerate( textTable_Content.items()):
                data =  Content[1][i] 
                string = str(data)
                T.write( string + '\t'                      )       
            T.write('\n')
        T.close()
        
        
        # write it into                            .bin ........................
        fo = open(featureTableFilename )                                                #Read featuretable from .txt generated from Nucleus Editor
        content  = fo.readlines()                                                      # the full data of the table 
        fo.close()
        content_Mat =  ([x.split('\t') for x  in content])   
        # ([x[len].split('\t') for x  in content_Mat])   
        featureVariableMat = content_Mat[1:len(content_Mat)]                                   # row[1:end]: extract the feature names
        featureNameLs = content_Mat[0]                                                 # row[0]:     FeatureNames 
        if featureNameLs[len(featureNameLs)-1] == '\n' :                               # if the last element of featureNameLs is "\n", delete the last one
            featureNameLs.remove('\n')  
            ([x.remove('\n')    for  x  in featureVariableMat])
            
        
        featureVariable_float = np.array(featureVariableMat,dtype = np.float32)
        featureVariableLs_float = np.reshape( featureVariable_float.T ,featureVariable_float.size)
        
        tableBinName = featureTableFilename.split('.')[0] + '.bin'
        # write into .bin file
        fout = open(tableBinName,'wb')               
        featureVariableLs_float.tofile(fout)                       
        fout.close()
        tableBinSize = os.path.getsize(tableBinName)        
        if tableBinSize != (len(featureVariableLs_float) *32 /8) :                      # the size of the binary file shall equal to ( #of objects * bitdepth) /8
            print ('[Error !] FeatureTable into binary image wrong')
        
        
        # write ICE file 
        ice.GenerateICE(Write_img_file_Loc)
        ice.GenerateICE(Write_img_file_Loc, FeatureTableOnly = True)    
        
        toc3 = time.clock()        
        
        print('Write into file use time(s) :', ( '{:.2f}'.format(toc3- toc2) ))


