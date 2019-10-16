# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 12:25:30 2017

@author: xli63
"""
from skimage.feature import blob_log
from skimage import util,segmentation,exposure,filters, morphology
from scipy import ndimage,stats,cluster,misc
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt



class Dataset_read_TBI(object):
    def __init__(self):        
        self.chs = [                                                                         #image color , name  ,  image color,     seed color
                ['DAPI'   , 'ARBc_FPI#6_Vehicle_20C_4110_C10_IlluminationCorrected_stitched.tif',  'R0_C10',[1,1,1], 'Red'  , [1,0,1]   ,[1,0,1]    ] ,
                ['NeuN'   , 'ARBc_FPI#6_Vehicle_20C_4110_C7_IlluminationCorrected_stitched.tif' ,  'R0_C7 ',[1,0.5,0], 'Green', [0.5,1,0] ,[0.5,1,0]  ],
				  ['S100'   , 'ARBc_FPI#6_Vehicle_20C_4110_C6_IlluminationCorrected_stitched.tif'  ,  'R0_C6' ,[1,1,0]   ],
				  ['IBA1'   , 'ARBc_FPI#6_Vehicle_20C_4110_C8_IlluminationCorrected_stitched.tif'  ,  'R0_C8 ',[1,0,0]   ],
				  ['APC' 	  , 'ARBc_FPI#6_Vehicle_20C_4110_C4_IlluminationCorrected_stitched.tif'  ,  'R0_C4' ,[0,1,0.5] ],
                ['Parv' 	, 'ARBc_FPI#6_Vehicle_20C_4110_C2_IlluminationCorrected_stitched.tif' ,  'R0_C2' ,[0,1,0.5] ],   
				                                                                                                    
				]                                                                                                    
        self.LoG_Paras = [ # min_blobradius, max blobradius, num_sigma , blob_thres , overlap   details in: http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log
                [ 'DAPI'   ,   [9           ,  20           , 35       ,    0.015   , 0.7] ] ,
#                        'DAPI'   ,   [ 12           ,40           ,  35      ,    0.015     ,  0.7 ]  ] ,
#               
#                [ 'NeuN'   ,  [ 30           ,50          ,  35      ,    0.015     ,  0.7 ]        
                [ 'NeuN'   ,   [ 22           ,32         ,  35      ,    0.02     ,  0.7 ]    ],   
                
#                 [ 'S100'   ,   [ 10           ,__       ,  __     ,    __    ,  __ ]  ],     # only need the size for segmentation
#                 [ 'IBA1'   ,  [ 10           ,__         ,  __     ,    __    ,  __ ]  ],     # only need the size for segmentation
#                 [ 'APC'   ,   [ 10           ,__         ,  __     ,    __    ,  __ ]  ],     # only need the size for segmentation
#                
                           ]  
        self.Other_Paras = [ # Bootstrap
                [ 'DAPI'   , [ True             ] ],
                [ 'NeuN'   , [ False             ] ],

                
                ]  
        self.Biomarker    = []
        self.FileName    = '! Unfounded'
        self.ChannelName = '! Unfounded'
        self.ColorCode   = '! Unfounded'
        self.ColorName   = '! Unfounded'
        self.ColorCode_border = '! Unfounded'
        self.ColorCode_seed =  '! Unfounded'
        self.LoG_Para =  '! Unfounded'
        self.Other_Paras =  '! Unfounded'
        
    def getFileName(self,Biomarker):
        for ch in self.chs:
            if ch[0] == Biomarker:
                self.FileName = ch[1]
        return self.FileName
    
    def getChannelName(self,Biomarker):
        for ch in self.chs:
            if ch[0] == Biomarker:
                self.ChannelName = ch[2]
        return self.ChannelName
    
    def getColorCode(self,Biomarker):
        for ch in self.chs:
            if ch[0] == Biomarker:
                self.ColorCode = ch[3]
        return self.ColorCode
    
    def getColorName(self,Biomarker):
        for ch in self.chs:
            if ch[0] == Biomarker:
                self.ColorName = ch[4]
        return self.ColorName
    
    def getColorCode_border(self,Biomarker):
        for ch in self.chs:
            if ch[0] == Biomarker:
                self.ColorCode_border = ch[5]
        return self.ColorCode_border
    
    def getColorCode_seed(self,Biomarker):
        for ch in self.chs:
            if ch[0] == Biomarker:
                self.ColorCode_seed = ch[6]
        return self.ColorCode_seed
    
    def getLoG_Paras(self,Biomarker):
        for LoG_Para in self.LoG_Paras:
            if LoG_Para [0] == Biomarker:
                self.LoG_Para = LoG_Para[1]
        return self.LoG_Para
    
    def getOther_Paras(self,Biomarker):
        for Other_Para in self.Other_Paras:
            if Other_Para [0] == Biomarker:
                self.Other_Para = LoG_Para[1]
        return self.Other_Paras