# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 14:18:32 2017

@author: xli63
"""

class blobFeatures:
    def __init__(self,obj):  # obj is the class of regionprop
        self.intensity_image = obj.intensity_image
        
        self.Tol_intensity = obj.intensity_image.sum()      # Tol_intensity 
        self.Avg_intensity = obj.mean_intensity  # mean_intensity 
        self.eccentricity = obj.eccentricity  
        self.orientation = obj.orientation 
        self.area  = obj.area  # Number of pixels of region.
        
        
        
bounding_box_volume
sum
mean
minimum
maximum
sigma
variance
surface_gradient
interior_gradient
surface_intensity
interior_intensity
intensity_ratio
convexity
radius_variation
surface_area
shape
shared_boundary
t_energy
t_entropy
inverse_diff_moment
inertia
cluster_shade
cluster_prominence
