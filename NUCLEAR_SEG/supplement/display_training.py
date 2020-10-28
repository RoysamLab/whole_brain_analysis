
# coding: utf-8

# In[4]:


import os, sys,glob
import random
import itertools
import colorsys
import skimage
import skimage.io as io
import numpy as np
from skimage.measure import find_contours,regionprops

   
from matplotlib import patches,	 lines
from matplotlib.patches import Polygon
import pandas as pd

import matplotlib.pyplot as plt


# In[5]:



def display_instances(image, boxes, masks, class_ids, class_names,
					  scores=None, title="",
					  figsize=(16, 16), ax=None,
					  show_mask=True, show_bbox=True,
					  colors=None, captions=None,mask_erosion_px = None, mask_linewidth =0.3):
	"""
	boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
	masks: [height, width, num_instances]
	class_ids: [num_instances]
	class_names: list of class names of the dataset
	scores: (optional) confidence scores for each box
	title: (optional) Figure title
	show_mask, show_bbox: To show masks and bounding boxes or not
	figsize: (optional) the size of the image
	colors: (optional) An array or colors to use with each object
	mask_linewidth (optional) float to control the width of mask border linewidth  (recommend 0.1 to 0.6)
	captions: (optional) A list of strings to use as captions for each object, 
				"[]",means not showing anything
	mask_erosion_px: (optional) integer of pixel to shink the mask	 (recommend 1 to 3)
	"""
	# Number of instances
	N = boxes.shape[0]
	if not N:
		print("\n*** No instances to display *** \n")
	else:
		assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

	# If no axis is passed, create one and automatically call show()
	auto_show = False
	if not ax:
		fig, ax = plt.subplots(1, figsize=figsize)
		auto_show = True

	# Generate random colors
	if colors is None:
		colors = random_colors(N)
	else:
		if len(colors) == 1:											# replicate item color to all instances
			colors_ls = []
			[colors_ls.append(colors) for i in range(0,N)]				#		  "r" to ["r","r","r"]
			colors = colors_ls

	# Show area outside image boundaries.
	height, width = image.shape[:2]
	ax.set_ylim(height + 10, -10)
	ax.set_xlim(-10, width + 10)
	ax.axis('off')
	ax.set_title(title)

	masked_image = image.astype(np.uint32).copy()
	for i in range(N):
		color = colors[i]

		# Bounding box
		if not np.any(boxes[i]):
			# Skip this instance. Has no bbox. Likely lost in image cropping.
			continue
		y1, x1, y2, x2 = boxes[i]
		if show_bbox:
			p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
								alpha=0.7, linestyle="dashed",
								edgecolor=color, facecolor='none')
			ax.add_patch(p)

		# Label
		if not captions:
			class_id = class_ids[i]
			score = scores[i] if scores is not None else None
			label = class_names[class_id]
			x = random.randint(x1, (x1 + x2) // 2)
			caption = "{} {:.3f}".format(label, score) if score else label
		else:
			if len(captions) == N:
				caption = captions[i]
			else:
				caption = captions[0]
		ax.text(x1, y1 + 8, caption,
				color='w', size=11, backgroundcolor="none")

		# Mask
		mask = masks[:, :, i]
		if show_mask:
			if mask_erosion_px is not None:
				mask = skimage.morphology.binary_dilation(mask, skimage.morphology.disk(mask_erosion_px))	# make mask thicker
			masked_image = apply_mask(masked_image, mask, color)

		# Mask Polygon
		# Pad to ensure proper polygons for masks that touch image edges.
		padded_mask = np.zeros(
			(mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
		padded_mask[1:-1, 1:-1] = mask
		contours = find_contours(padded_mask, 0.5)
		for verts in contours:
			# Subtract the padding and flip (y, x) to (x, y)
			verts = np.fliplr(verts) - 1
			p = Polygon(verts, facecolor="none", edgecolor=color,linewidth = mask_linewidth)
			ax.add_patch(p)
	ax.imshow(masked_image.astype(np.uint8))
	if auto_show:
		plt.show()
	
	return fig


# In[6]:


def random_colors(N, bright=True):
	"""
	Generate random colors.
	To get visually distinct colors, generate them in HSV space then
	convert to RGB.
	"""
	brightness = 1.0 if bright else 0.7
	hsv = [(i / N, 1, brightness) for i in range(N)]
	colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
	random.shuffle(colors)
	return colors
def apply_mask(image, mask, color, alpha=0.5):
	"""Apply the given mask to the image.
	"""
	for c in range(3):
		image[:, :, c] = np.where(mask == 1,
								  image[:, :, c] *
								  (1 - alpha) + alpha * color[c] * 255,
								  image[:, :, c])
	return image


# In[ ]:



def saveAll(sample_root, multiclass = True):
	for sample_fn in os.listdir(sample_root):
		sample_dir = os.path.join(sample_root,sample_fn)
		print("sample_dir=",sample_dir)
		image_dir=os.path.join(sample_dir,"images")
		mask_dir=os.path.join(sample_dir,"masks")
		image = io.imread( os.path.join(image_dir,os.listdir(image_dir)[0]))[:,:,:3]
		boxes =[]
		masks=[]

		if multiclass is True: # multiclass
			class_ids = []
			class_names=["None","NeuN","IBA1","Olig2","S100","RECA1"]
			class_csv_dir = next(glob.iglob(os.path.join(mask_dir, "*.csv")))
			class_csv = pd.read_csv(class_csv_dir,index_col="id")
			# Read mask files from .png image
			masks = []
			for f in next(os.walk(mask_dir))[2]:
				if f.endswith(".png"):
					m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
					if m.sum() > 0:
						m_obj=regionprops(m*1)[0]
						boxes.append(m_obj.bbox)
						masks.append(m)		  
						obj_id = f.split("-")[1].split(".png")[0]
						class_ids.append(class_csv["class"][int(obj_id)])
			if masks==[]:
				masks=np.zeros_like(m)
			masks = np.stack(masks, axis=-1)

			fig=display_instances(image,np.array(boxes),masks,np.array(class_ids,dtype=int),class_names,
								title=f.split("-")[0], show_bbox=False, show_mask=False,mask_linewidth = 2.5)

			fig.savefig(os.path.join(sample_dir,f.split("-")[0]+"_annotation.png"))
		else:	# single class
			print ("single class")
			class_names=["None","Nucleus"]
			# Read mask files from .png image
			masks = []
			class_ids = []
			for f in next(os.walk(mask_dir))[2]:
				if f.endswith(".png"):
					m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
					if m.sum() > 0:
						m_obj=regionprops(m*1)[0]
						boxes.append(m_obj.bbox)
						masks.append(m)		  
						obj_id = f.split("-")[1].split(".png")[0]
						class_ids.append(1)
			if masks==[]:
				masks=np.zeros_like(m)
			masks = np.stack(masks, axis=-1)

			fig=display_instances(image,np.array(boxes),masks,np.array(class_ids,dtype=int),class_names,
								title=f.split("-")[0], show_bbox=False, show_mask=False,mask_linewidth = 2.5)

			fig.savefig(os.path.join(sample_dir,f.split("-")[0]+"_annotation.png"))			


# In[ ]:


if __name__ == '__main__':
	import matplotlib
#	  # Agg backend runs without a display
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	
	import argparse,time
	# Parse command line arguments
	parser = argparse.ArgumentParser(
		description='',
		formatter_class=argparse.RawTextHelpFormatter)
		
	parser.add_argument('-i','--imgRoot', required=False,
						metavar = "/path/to/dataset/",
						default = None,
						help='Root directory of images and masks to read')
	parser.add_argument('-m','--multiclass', required=False,
						metavar = "/path/to/dataset/",
						default = True,
						help='Root directory of images and masks to read')
	# sample_root="/brazos/roysam/xli63/exps/SegmentationPipeline/mrcnn_Seg/datasets/50_plex/multiplex/whole/train/"
	
	args = parser.parse_args()
	tic			= time .time()
	saveAll(args.imgRoot,args.multiclass)
	toc			= time .time()
	print ( "total time is (h) =", str((toc-tic)/3600))
	

