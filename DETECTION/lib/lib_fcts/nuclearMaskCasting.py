# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 14:50:07 2017

@author: xli63
"""
def CastNuclearMaskonOtherChannels(Images, labels_soma, seeds_marker ,Write_img_file_Loc)
                                             # biomarker name is inside the fileName                                                       
    for Image in  Images.findall('Image'):    
        #    biomarker = imreadName.split('__')[0]                                                  # biomarker name is inside the fileName    
        biomarker = Image.get('biomarker')
        if biomarker != 'NONE' :       #Implement LoG on whole image  only for DAPI channel  
            imreadName = Image.find('FileName').text
            image = io.imread( checkPath (Read_img_file_Loc+'\\' + imreadName) )
            if trywholeImg == False:
                image = image[cropRange[0][0]:cropRange[0][1],cropRange[1][0]:cropRange[1][1]]     # left corder
            
            ChannelName = '[' +biomarker+']'    
            
            imWriteName = ChannelName + '_8bit.tif'
            cv2.imwrite (checkPath (Write_img_file_Loc + '\\' + imWriteName), skimage.img_as_ubyte(image))                      # display for 8 bits
#            
#            if CropFolderName in ['R2Soma']:
#                saveDisp = False  # whole image no need to save for display
#            else:
#                saveDisp = True          
            
    ##        # read seeds from deep learning table     
    #        print (ChannelName)
            writeTifandBin(Write_img_file_Loc, ChannelName + 'soma' , image, labels_soma , seeds_marker, display = False, saveDisp = saveDisp,writeBin = True)
            PropertyTables[(biomarker +'_somaMasked') ] = measure.regionprops(labels_soma,intensity_image=image)                  # storage the properties e.g ID,area  of each componentes (labelled regions)
            
#            writeTifandBin(Write_img_file_Loc, ChannelName + 'nucleus' , image, labels_nucleus , seeds_marker, display = False, saveDisp = saveDisp,writeBin = False)
#            PropertyTables[(biomarker +'_nucleusMasked') ] = measure.regionprops(labels_nucleus,intensity_image=image)                  # storage the properties e.g ID,area  of each componentes (labelled regions)

#            writeTifandBin(Write_img_file_Loc, ChannelName + 'plasma' , image, labels_plasma , seeds_marker, display = False, saveDisp = saveDisp,writeBin = False)
#            PropertyTables[(biomarker +'_plasmaMasked') ] = measure.regionprops(labels_plasma,intensity_image=image)                  # storage the properties e.g ID,area  of each componentes (labelled regions)

    
#    for AVGonly in [1,0]:
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
#        textTable_Content[maskName + '__Area'         			] = np.zeros(NumOfObj,dtype = np.uint32)
        textTable_Content[maskName + '__Tol_intensity'			] = np.zeros(NumOfObj,dtype = np.float32)
        textTable_Content[maskName + '__Avg_intensity'			] = np.zeros(NumOfObj,dtype = np.float32)
#        textTable_Content[maskName + '__Eccentricity'   		] = []
#        textTable_Content[maskName + '__Equivalent_diameter '	] = []
        for obj in  PropertyTable[1]: # the contents  name
                label_ID = obj.label
#                textTable_Content[maskName + '__Area'         			][label_ID-1]  = np.uint32(obj.area)     # volum (area :1 int Number of pixels of region.)
                textTable_Content[maskName + '__Tol_intensity'			][label_ID-1]  = np.float32(obj.intensity_image.sum() )      # Tol_intensity 
                textTable_Content[maskName + '__Avg_intensity'			][label_ID-1]  = np.float32(obj.mean_intensity)   # mean_intensity 
#                textTable_Content[maskName + '__Eccentricity'   		    ].append ( ( np.float32(obj.eccentricity) )              )      # eccentricity
#                textTable_Content[maskName + '__Equivalent_diameter '	].append ( ( np.float32(obj.equivalent_diameter) )       )      # equivalent_diameter
	
    datasetName = xmlreadName.split('Dat')[0]
    #   write properties of components      into txt.................  for ICE
    featureTableFilename = checkPath ( Write_img_file_Loc+ '\\' + datasetName + '_FeatureTable.txt' )
    T = open(featureTableFilename, 'w')        # create a table txt to write
    #write header
    T.write('ID' + '\t' + 'centroid_x' + '\t' + 'centroid_y' + '\t' )
    for Content in textTable_Content.items():
    #    print (Content[0])
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
#            if i != (len( textTable_Content.items()) -1):
#                T.write( string + '\t'                      )       
#            else:
#                T.write( string                     )    1
        T.write('\n')
    T.close()
    
#    ##   write properties of components      into txt (comma version).................   for FCS   string
#    featureTableFilename_FCS = Write_img_file_Loc+ '\\' + datasetName + '_FeatureTable(commaforFCS).txt'
#    T = open(featureTableFilename_FCS, 'w')        # create a table txt to write
#    #write header
#    T.write('ID' + ',' + 'centroid_x' + ',' + 'centroid_y' + ',' )
#    for Content in textTable_Content.items():
#    #    print (Content[0])
#        T.write(Content[0] + ',' ) 
#    T.write('\n')      
#    #write data
#    for i in range(len(textTable_ObjID['ID'])):
#        T.write(str( textTable_ObjID['ID'][i]        )    + ','                 )      # ID
#        T.write(str( textTable_ObjID['centroid_x'][i] )   + ','                 )      # centroid_x
#        T.write(str( textTable_ObjID['centroid_y'][i] )  + ','                 )      # centroid_y    
#        for j, Content in enumerate( textTable_Content.items()):
#            data =  Content[1][i] 
#            string = str( "%.3f" % data)
#            if j != (len( textTable_Content.items()) -1):
#                T.write( string + ','                      )      
#            else:
#                T.write( string                     )      
#        T.write('\n')
#    T.close()
    
    
    
    
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
        
    #     export the real feature Values data, generate into bin file 
    
    featureVariable_float = np.array(featureVariableMat,dtype = np.float32)
    featureVariableLs_float = np.reshape( featureVariable_float.T ,featureVariable_float.size)
    
    tableBinName = featureTableFilename.split('.')[0] + '.bin'
    # write into .bin file
    fout = open(tableBinName,'wb')               
    featureVariableLs_float.tofile(fout)                       
    fout.close()
    tableBinSize = os.path.getsize(tableBinName)        
    ##print (tableBinSize)
    if tableBinSize != (len(featureVariableLs_float) *32 /8) :                      # the size of the binary file shall equal to ( #of objects * bitdepth) /8
        print ('[Error !] FeatureTable into binary image wrong')
    