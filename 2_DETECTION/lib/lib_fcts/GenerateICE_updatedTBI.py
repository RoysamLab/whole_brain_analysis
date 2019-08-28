# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:47:07 2017ele

@author: xli63
"""
'''
%Input:
% [featuretable.txt] Feature table from NucleusEditor ( temp use tableClassifiedValidatedIbaS100NeunApcReca1GAD67ParvCalrCC3.txt )
% [mask_bin.tif] segmenation mask from NucleusEditor (temp use 'NeuN_Mask_Bin.tif' )
%           Background=0; borders =1; foreground = cell ID corresponding to  object ID
% [original image]  illuminaionCorrectedImage (temp  % use'4110_C1_IlluminationCorrected_stitched.tif'   )
% [externel ..def.xml ]file to get the channel information #
%Output:
% [.ice] XMLfile
% [featuretable.bin]
% [featuretable.bin]
'''

# Generate ICE.py 
import sys
#import ice_api as api  # call the python data structure generated from xsd
import os
#from Tkinter import *
import tkinter
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import tostring, Element, SubElement, Comment
from xml.dom import minidom
from PIL import Image as Img
import struct
import numpy as np
#from array import array
#
##   Ask to open the tiff file that you want to read 
#root = Tkinter.Tk()
#root.filename = tkFileDialog.askopenfilename(initialdir = '/', title = "Select file", filetypes = (("tiff files","*.tif"),("all files","*.*")))
#inputFileDir = os.path.dirname(root.filename)
def GenerateICE(inputFolder, FeatureTableOnly = False):
        
    def prettify(elem):
        """Return a pretty-printed XML string for the Element.
        """
        rough_string = tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")
    
    ###################################  Read input files ###########################
    
    #ParentLoc = os.path.dirname(os.getcwd())
    #inputFolder = os.getcwd() +'\\' + 'ICE_Result\\'
    allFileNames = os.listdir(inputFolder)                               #list all files from the input Directory
    inputFolder = inputFolder + '\\'
    #allFileNames = os.listdir(inputFileDir)                                       #list all files from the input Directory♥
    actualImageNames = []
    maskImageNames = []
    DAPI_masks_fileNames = []
    actualImageNames_ExceptforDAPI = []
    #defXMLNames = []
    for fileName in allFileNames:
        if (fileName.split('.')[1] == 'bin'):                 ## Mask.bin
            if 'Mask' in fileName:
                maskImageNames.append(fileName)
        #        if fileName.split('_')[0] == 'DAPI' :                                   # DAPI ...Mask
        #            DAPI_masks_fileNames.append(fileName)
            elif 'FeatureTable' in fileName or 'DatesetDefinition' in fileName:
                featureTableFilename_bin = fileName             # find bin. txt  
        if (fileName.split('.')[1] == 'tif'):
            if fileName.split('_')[1]=='8bit.tif' :
                actualImageNames.append(fileName)
    
    
        
    
    #    if  (fileName[len(fileName)-8:len(fileName)] == '_def.xml'):                ## defination.xml
    #        defXMLNames.append(fileName)
    
    
    # read feture table
    #inputFolder =  (r'D:\research in lab\NIHIntern(new)\RebeccaCode\Change2ICE\ICE_Result\\')
    # Read the external feature table.txt
    featureTableFilename = featureTableFilename_bin.split('.')[0] + '.txt'
    fo = open(inputFolder +  featureTableFilename )                                                #Read featuretable from .txt generated from Nucleus Editor
    content  = fo.readlines()                                                      # the full data of the table 
    fo.close()
    content_Mat =  ([x.split('\t') for x  in content])   
    # ([x[len].split('\t') for x  in content_Mat])   
    
    
    featureVariableMat = content_Mat[1:len(content_Mat)]                                   # row[1:end]: extract the feature names
    
    featureNameLs = content_Mat[0]                                                 # row[0]:     FeatureNames 
    if featureNameLs[len(featureNameLs)-1] == '\n' :                               # if the last element of featureNameLs is "\n", delete the last one
        featureNameLs.remove('\n')  
        ([x.remove('\n')    for  x  in featureVariableMat])
        
    #     export the real feature Values data, generate into bin file 
    
    featureVariable_float = np.array(featureVariableMat,dtype = np.float32)
    featureVariableLs_float = np.reshape( featureVariable_float.T ,featureVariable_float.size)
    
    #s = struct.pack('f'*len(featureVariableLs_float), *featureVariableLs_float)   # convert the float into binary datastructure
    
    channelNames = []
    channelTypes = []
    #for defXMLName in defXMLNames :                                                  
                
    #####################################  Generateing ICE  ##########################
    rootET = Element('ICEFormat')
    rootET.set('xmlns','http://www.isac-net.org/std/ICEFormat/1.0/ice')
    rootET.set('xmlns:xsi','http://www.w3.org/2001/XMLSchema-instance')
    rootET.set('xsi:schemaLocation','http://www.isac-net.org/std/ICEFormat/1.0/ \n http://flowcyt.sf.net/ice/ICE.xsd')
    rootET.set('version','1.0')
    comment = Comment('Description of the contents of the ICEFormat Data Directory file')
    rootET.append(comment)
    
    #### Parsing Channel Definitions    
    #ChannelDefinitions = SubElement(rootET, 'ChannelDefinitions')
    #for channelInd , (channelName,  Biomarker ) in enumerate(zip (channelNames,Biomarkers)) :
    #    Channel = SubElement(ChannelDefinitions, 'Channel')
    #    Channel.set('Id',channelName)
    #    Channel.set('Description',Biomarker)
    #    
    #Additional = SubElement(Channel, 'Additional')
    #custom = SubElement(Additional, 'custom')
    #custom.text= 'metadata'
    
    
    #### Doubts:  Segmentation ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #SegmentationDefinitions = SubElement(rootET, 'SegmentationDefinitions')
    #Segmentation = SubElement(SegmentationDefinitions, 'Channel')
    #Segmentation.set('Id','NucleusEditor')
    #Segmentation.set('Description','Yousef_s Method')
    #Additional = SubElement(Segmentation, 'Additional')
    #custom = SubElement(Additional, 'custom')
    #custom.text= 'metadata'
    
    ### FeatureDefinitions share among all datasets
    FeatureDefinitions = SubElement(rootET, 'FeatureDefinitions')         
    
    #FeatureValue = SubElement(FeatureDefinition, 'FeatureValue')
    #Primitive = SubElement(FeatureValue,'Primitive')
    PrimitiveFIDs = []
    for FID, featureName in enumerate(featureNameLs):      #featureNameList read from .txt 
        if len(featureName)>1:
            FeatureDefinition = SubElement(FeatureDefinitions, 'FeatureDefinition')       
            InfoFloat = SubElement(FeatureDefinition, 'InfoFloat')
            Description = SubElement(InfoFloat, 'Description')
            Description.text = featureName                # call for featureName
            ID = SubElement(InfoFloat, 'ID')
            ID.text = 'F0'+str(FID)
            PrimitiveFIDs.append(ID.text)
            BitDepth = SubElement(InfoFloat, 'BitDepth')
    #        BitDepth.text = str(32)     
            BitDepth.text = str(32)     
    
    
     ###   Dataset                 different animals are in different dataset
    Dataset = SubElement(rootET, 'Dataset')                
    Dataset.set('Info', '50CHN R2 somma')
    
    ##   featureTable                                                                
    Metadata = SubElement(Dataset, 'Metadata')
    NumberOfObjects = SubElement(Metadata,'NumberOfObjects')                       
    NumberOfObjects.text = str(len(featureVariableMat))                       # Number of Objects = Number of Rows ub featureVariableLs
    Custom = SubElement(Metadata,'Custom')
    DataGeneratedBy = SubElement(Custom,'DataGeneratedBy')
    Make = SubElement(DataGeneratedBy,'Make')
    Make.text = 'WatershedSegmentation'
    
    
        ## FeatureValues       # define the featurevalue save file
    FeatureValues = SubElement(Dataset, 'FeatureValues')
    ### feature Value for actual features
    FeatureValue  = SubElement(FeatureValues, 'FeatureValue')
    Primitive =  SubElement(FeatureValue, 'Primitive')
    URL = SubElement(Primitive,'URL')
    featureTableBinname = featureTableFilename.split('.')[0] + ('.bin')
    URL.text = ('file://'+ featureTableBinname)        # get the file Name of Image        
    for PrimitiveFID in PrimitiveFIDs:                    # read ele 
            FeatureID = SubElement(Primitive, 'FeatureID')
            FeatureID.text = PrimitiveFID            
    ## images
    CompositeImages = SubElement(Dataset, 'CompositeImages')
    CompositeImages.append(Comment('Association with channels definition'))
    
    ImageIDs_others = []
    Biomakers_others = []
#    print(actualImageNames)
    actualImg = Img.open(inputFolder + actualImageNames[0])    

    for ImgID , actualImageName in enumerate(actualImageNames):
        Image = SubElement(CompositeImages,'Image')
        ID = SubElement(Image,'ID')
        ID.text = 'Img_00' + str(ImgID)                                             # only keep the name part before extention as the ID 
        actualImg = Img.open(inputFolder + actualImageName)                          # Read Image
        URL = SubElement(Image,'URL')
        URL.set('url','file://'+ actualImageName)                      # get the file Name of Image    
        Width = SubElement(Image,'Width')
        Width.text = str( actualImg.size[0])
        Height = SubElement(Image,'Height')
        Height.text = str(actualImg.size[1])
        
        biomarkerName = actualImageName.split('_')[0]                               # prepare for InfoCompositeImage
        Image.set('biomarker',biomarkerName) 
        if biomarkerName == '[DAPI]':                                              #  
            ImageID_DAPI = ID.text
        else:
            ImageIDs_others.append( ID.text )                                      # for use later in InfoCompositeImage
    
    
    
    if FeatureTableOnly == False:
        ##   Masks    [read the mask image .bin external file]
        Masks = SubElement(Dataset, 'Masks')
        Masks.append(Comment('Association with segmentation and feature value defination'))    
        
        MaskIDs_Nucleus =[]
        MaskIDs_others =[]
        for MaskID , maskImgName in enumerate(maskImageNames):
            #individual Mask images
            Mask = SubElement(Masks,'Mask')
            ID = SubElement(Mask,'ID')
            ID.text = 'Mask_00' + str(MaskID)                                                # mask ID correspond to image and channel
        #    MaskID.text = ID.text                                                      # correspond to ImageID in InfoCompositeImage
            URL = SubElement(Mask,'URL')
            URL.text = ('file://'+  maskImgName.split('.')[0]+'.bin')                 # get the file Name of MaskImage (Should be .bin file)
              # Read Image size
            Width = SubElement(Mask,'Width')
            Width.text = str( actualImg.size[0])
            Height = SubElement(Mask,'Height')
            Height.text = str(actualImg.size[1])
            BitDepth = SubElement(Mask, 'BitDepth')
            BitDepth.text = str(32)                # unsigned integer 32 bit
            
            biomarkerName = maskImgName.split(']')[0] +']'                                    # prepare for InfoCompositeImage
        #    Mask.set('biomarker',actualImageName.split('_')[0]) 
            maskType = maskImgName.split(biomarkerName)[1].split('_')[0]
            if biomarkerName in ['[DAPI-NeuN]','[NeuN+Parvalbumin]']:                                                  # nuclues Mask, generate for all channels
                MaskIDs_Nucleus.append(ID.text)
            else:                                                                            # morphological Mask  # might us in the future        generate for all other channels
                MaskIDs_others.append(ID.text)
                
            
        
    
                
        # exit dataset-----------------------------
        
        
        
        # other feature definition for all 
        #Description = SubElement(InfoCompositeImage, 'Description')
        #D
        #esciption.text = str(channelType)                                           # may change is to channelType
        #ID = SubElement(InfoCompositeImage, 'ID')
        #ID.text = 'COMP_IMG_00'+str(channelInd)
        #
        
        CompositeImageFIDs = []        # save the CompositeImageFIDs for FeatureValue stack holder
        # Generate Composite Image for DAPI channel and DAPI masks (nucleus_Mask, wholeCell_Mask and cytoplasm_Mask)
        COMP_ID = 1
        for Image in  CompositeImages.findall('Image'):                                 # for each channel Img
            ImgID = Image.find('ID').text                                               # image ID 
            ImgBiomaker = Image.get('biomarker')                                        # biomarker (channel Name)   '[]'
            
            for Mask in  Masks.findall('Mask'):                                        # for each mask   (find previous definition)
                MaskID = Mask.find('ID').text      
                
                if MaskID in MaskIDs_Nucleus:                                   # generate(nucleus_Mask, wholeCell_Mask and cytoplasm_Mask) for all channels
                    MaskID_Nucleus = MaskID
                    
                    FeatureDefinition = SubElement(FeatureDefinitions, 'FeatureDefinition')       
                    InfoCompositeImage = SubElement(FeatureDefinition, 'InfoCompositeImage')              # will define  later after we read Image and masks
                    Description = SubElement(InfoCompositeImage, 'Description')
                 
                    MaskDescription_Nucleus = Mask.find('URL').text.split(']')[1] 
                    MaskDescription_Nucleus = MaskDescription_Nucleus.split('.')[0]               # get mask type  from file name ,e.g."nuclues_Mask"
                    Description.text = ImgBiomaker + ' with ' + MaskDescription_Nucleus                           # [Channel] in mask
                    
                    ID = SubElement(InfoCompositeImage, 'ID')
                    ID.text = 'F0_COMP_'+ str(COMP_ID)                                              #create the compond image ID
                    COMP_ID = COMP_ID + 1
                
                    CompositeImageFIDs.append(ID.text)            
                    ImageID = SubElement(InfoCompositeImage, 'ImageID')                             # will define later after we read Image
                    ImageID.text = ImgID
                    
                    MaskID = SubElement(InfoCompositeImage, 'MaskID')                               # will define later after we read mask
                    MaskID.text = MaskID_Nucleus
                    
                elif MaskID in MaskIDs_others:                                           # assciate its morphological Mask
                    MaskBiomaker =  Mask.find('URL').text.split('[')[1] + '['                 # get channel name from file name ,e.g."[S100]"
                    MaskBiomaker = MaskBiomaker.split(']')[0] + ']'
                    if MaskBiomaker == ImgBiomaker:                                         # only associate the morphology mask with the corresponding channel
                        MaskID_other =  MaskID
                        FeatureDefinition = SubElement(FeatureDefinitions, 'FeatureDefinition')       
                        InfoCompositeImage = SubElement(FeatureDefinition, 'InfoCompositeImage')              # will define  later after we read Image and masks
                        Description = SubElement(InfoCompositeImage, 'Description')
                        
                        MaskDescription_Nucleus = Mask.find('URL').text.split(']')[1] 
                        MaskDescription_Nucleus = MaskDescription_Nucleus.split('.')[0]               # get mask type  from file name ,e.g."morpholgoical_Mask"
                        Description.text = ImgBiomaker + ' with ' + MaskDescription_Nucleus                          # [Channel] in mask
                        
                        ID = SubElement(InfoCompositeImage, 'ID')
                        ID.text = 'F0_COMP_'+ str(COMP_ID)                                              #create the compond image ID
                        COMP_ID = COMP_ID + 1
                    
                        CompositeImageFIDs.append(ID.text)            
                        ImageID = SubElement(InfoCompositeImage, 'ImageID')                             # will define later after we read Image
                        ImageID.text = ImgID
                        
                        MaskID = SubElement(InfoCompositeImage, 'MaskID')                               # will define later after we read mask
                        MaskID.text = MaskID_other
                            
        
              
        # supplemetry for feature Values of Composite Image (because the ID have to be define previousely)
        ### feature Value for CompositeImage
        FeatureValue = SubElement(FeatureValues, 'FeatureValue')
        CompositeImage =  SubElement(FeatureValue, 'CompositeImage')                   # for stack holder
        for CompositeImageFID in CompositeImageFIDs:                          # read ele 
                FeatureID = SubElement(CompositeImage, 'FeatureID')
                FeatureID.text = CompositeImageFID 
        #
        ##
        ### write into xml file
    #    print (prettify(rootET))    # display in screen
        
        xml_fileName = 'TBIR1.ice'
    else:  # FeatureTableOnly = False
        xml_fileName = 'TBIR1_FeatureTableOnly.ice'

    xml_file = open(inputFolder + xml_fileName,'w')    
    xml_file.write(prettify(rootET))
    xml_file.close()

