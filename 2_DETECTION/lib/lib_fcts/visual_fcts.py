import matplotlib.patches as mpatches
import cv2 ,os
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




def warningIgnore():
    warnings.filterwarnings("ignore")

def checkPath(loc, mkdir = False):
    new_loc = loc

    if sys.platform == "linux" or sys.platform == "linux2":
        if "\\" in loc:
          locLinux = loc.replace("\\", "/")
          new_loc =  locLinux          
    elif sys.platform == "windows" :
        if "/" in loc:
          locLinux = loc.replace("/", "//")
          new_loc =  locLinux
    else:
        new_loc =  loc
    
    if  mkdir == True:
        if os.path.isdir(loc) == False:
            os.makedirs(loc)
    return new_loc
        


def imadjust16bit(img):
    img = img/ (img.max())    # float
    img = skimage.img_as_uint(img)    # 16 bit
    return img


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


##''' input image : DAPI + HISTONE (pixel summation) '''   
def imSummation (img1,img2) :  # save size  
    imSum = np.add(img1, img2)
    imSum = imSum/ imSum.max()            # float
    imSum = skimage.img_as_uint(imSum)   # change to 16bit 
    
    return imSum
##DAPI_p_HISTONE = np.add(channels_images[0] , channels_images[1])
##DAPI_p_HISTONE =  DAPI_p_HISTONE/DAPI_p_HISTONE.max()


def convertCirclelabel(label , blob , seeds_marker):
    imgLabel = np.zeros_like(label)
    
    x, y = np.indices(label.shape)
    for i in  range (blob.shape[0]):    
        x_i = blob[i,0]
        y_i = blob[i,1]
        r_i =(blob[i,2]) 
        r_i_adjusted = r_i * 3.6/np.log(r_i) 
#        rr, cc = skimage.draw.circle( *list(blob[i,:]))
        rr, cc = skimage.draw.circle( x_i,y_i,r_i_adjusted)
        r_bdId =  rr< label.shape[0]                             # fixin the boundary of image
        c_bdId =  cc< label.shape[1] 
        bdId = np.logical_and(r_bdId,c_bdId)   

        imgLabel[rr[bdId] , cc[bdId] ] = 1
        
    D = ndimage.distance_transform_edt(imgLabel)    
    
    circularLabel = morphology.watershed(-D, seeds_marker,  mask = imgLabel)       
    
#    plt.figure(),plt.imshow(circularLabel), plt.title ('circularLabel')

    
    return circularLabel


#    ellipsoidlabel = convertEllipsoidlabel(labels_Neuron , blobs_Neuron , seeds_marker_Neuron)

def convertEllipsoidlabel (label, blob, seeds_marker, onlymask= False):
    imgLabel = np.zeros_like(label)
    
#    plt.figure(),plt.imshow(label == 2883)
    
    x, y = np.indices(label.shape)
    for obj in   measure.regionprops(label):    
        
        r,c = obj.centroid
        
        if obj.minor_axis_length==0 or obj.major_axis_length==0:  # square
            imgLabel[obj.coords[:,0], obj.coords[:,1]] =1      
        #        rr, cc = skimage.draw.circle( *list(blob[i,:]))
        else:
            r_radius = obj.minor_axis_length    /2
            c_radius = obj.major_axis_length    /2
        
            r_radius_adjusted = r_radius * 3/np.log(r_radius) 
            c_radius_adjusted = c_radius * 3/np.log(c_radius) 
            ori = obj.orientation 
            
            rr, cc = skimage.draw.ellipse(int(r), int(c), int(r_radius_adjusted), int( c_radius_adjusted), rotation= ori)
            r_bdId =  rr< label.shape[0]              # fixin the boundary of image
            c_bdId =  cc< label.shape[1] 
            bdId = np.logical_and(r_bdId,c_bdId)   
    
            imgLabel[rr[bdId] , cc[bdId] ] = 1
        
    if  onlymask == True:   # only output elipsoide mask mask
        return imgLabel
    else:        
        D = ndimage.distance_transform_edt(imgLabel)    
        
        ellipsoidLabel = morphology.watershed(-D, seeds_marker,  mask = imgLabel)       
        
    #    plt.figure(),plt.imshow(ellipsoidLabel), plt.title ('ellipsoidLabel')
    
        
        return ellipsoidLabel

def extractFileNamesforType(dir_loc, fileExt):   # dir , '.tif'
    readNames =    os.listdir (dir_loc)
    types = np.array( [ ([ x.find(fileExt)>0 for x in  readNames ]) ] , dtype = int)   # only extract extension .tif
    typesIDs = np.array  (  np.where(types > 0)[1] )
    fileNames = []
    ([fileNames.append(readNames[i]) for i in typesIDs])    
    return fileNames
           
def visualize(img,labels = [],seeds_marker = [],seeds_marker_2 = []):  
# img: grayscale images (only to extract the size)
# labels :  (segmentation image)
#    labeled image: 0: backgourd, 
#                   1,2,3... : components ID  
#
# seeds_markered: (seeds image, default : disk radius size 3)                     
#    seeds_markered image: 0: backgourd, 
#                   1,2,3... : seeds ID
# e.g. 
# seeds_marker, labels, __,__ = CellSegmentationfcts_hierarchy.watershedSegmentation( IMG, blobs=[])
# img_disp = visualize(IMG ,label , seeds_marker,seeds_marker_2=[] )    
# plt.imshow(img_disp)

	
    img_adj = skimage.img_as_float(img)
    img_adj =  exposure.rescale_intensity(img_adj)
    img_disp = cv2.merge((img_adj,img_adj,img_adj))
	
    if labels != []:
        border = segmentation.find_boundaries(labels)
        img_disp[border==1]=[0,0.5,0]                                             # watershed mask border :green
		
    if seeds_marker != []:
        img_disp[seeds_marker > 0]=[0.5,0,0]                                      # detected seeds        :red
       
    if seeds_marker_2 != []:
        img_disp[seeds_marker_2 > 0]=[0,0,1]                                      # detected seeds        :blue
   
    return img_disp


def writeTifandBin(Write_img_file_Loc,fileNm, IMG, label, seedsmarker, displayonly = True): 
    img_disp = visualize(IMG ,label ,  seedsmarker)    
    rgb = skimage.color.label2rgb(label,image=None, bg_label=0, bg_color=(0, 0, 0))

    if displayonly == True:                # only display,no write
        plt.figure(),plt.imshow(img_disp)
        plt.figure(),plt.imshow(rgb)
        return []
    else:                                 # write in to file, no display
        tiffileNm  = (fileNm + '_Mask_DisplayBorder')     
        cv2.imwrite (Write_img_file_Loc + '\\Outputs\\' + tiffileNm +' .tif' ,skimage.img_as_ubyte(img_disp))     # display for 8 bit        
        
        tiffileNm  = (fileNm + '_Mask_DisplayLabel')      
        cv2.imwrite (Write_img_file_Loc + '\\Outputs\\' + tiffileNm +' .tif' ,skimage.img_as_ubyte(rgb))          # display for 8 bit
        
        # write labelleds image masks
        maskfileNm = (fileNm + '_Mask')           
        labelLocFileName = Write_img_file_Loc + '\\Outputs\\' + maskfileNm +'.out'
        np.savetxt(labelLocFileName, label, fmt='%d',delimiter=',')   # X is an array        
        # write labelled image masks into bins
        fout = open(Write_img_file_Loc + '\\Outputs\\' + maskfileNm + '.bin','wb')                            # 4byte =32 bit       
        labels_uint32 = np.uint32(label)                                                                     # change it to uint 32 bit with 4 bytes             
        labels_uint32.tofile(fout)                                                                               # write into .bin file
        fout.close()
        return labelLocFileName
    
