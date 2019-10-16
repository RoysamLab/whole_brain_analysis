# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 15:46:40 2018

@author: xli63
"""

import os
import numpy as np
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import tostring, Element, SubElement
from xml.dom import minidom

def readXML(xmlreadName, Searchbiomarker,key):
    if ('XLSX.xml' in xmlreadName) ==False:
        print ('not a .xml file')

    tree = ET.parse(xmlreadName)
    root = tree.getroot()
    Images = root[0]
        
    def getAttri(Searchbiomarker,key):    
        output = []
        for Image in  Images.findall('Image'):                            # take the current animal 
            bioMarker = Image.get('biomarker')
    #        print (bioMarker)
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
    
    val = getAttri(Searchbiomarker,key)
    
    return val