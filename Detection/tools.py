import numpy as np
from skimage import transform, feature

###########################################
"""
author: Kim-Louis Simmoteit

This module contains helpful functions for
calculating stuff.
"""
###########################################

SIZE_X = 64		#size of final image in x direction 64
SIZE_Y = 64		#size of final image in y direction 64

#Intersection over Union
def overlap(box1, box2):
	area1 = (box1[2]-box1[0]+1)*(box1[3]-box1[1]+1)
	area2 = (box2[2]-box2[0]+1)*(box2[3]-box2[1]+1)
	
	#intersection region
	miny = max(box1[0], box2[0])
	minx = max(box1[1], box2[1])
	maxy = min(box1[2], box2[2])
	maxx = min(box1[3], box2[3])
	
	intersect = max(0, maxy-miny+1)*max(0, maxx-minx+1)
	total_area = float(area1+area2-intersect) #don't count area 2 times
	
	return intersect/total_area


#Calculate HOG-Descriptor from an (part of an) image
def HogDescriptor(cropped):
	resized = transform.resize(cropped, (SIZE_X, SIZE_Y), anti_aliasing=True, mode="constant")
	desc = feature.hog(resized, pixels_per_cell=(6,6), cells_per_block=(2,2), visualize=False, block_norm="L1", transform_sqrt=True, multichannel=True)
	#desc = np.concatenate((resized[:,:,0].flatten(),resized[:,:,1].flatten(),resized[:,:,2].flatten()))
	return desc #desc.flatten()

#check if box fulfills rough sign criterium (70%<aspect-ratio<130% & min_width/height=20, area/im_area < 80)
#returns False if box is accepted
def filter_box(box, im_area):
	height = float(box[2]-box[0]+1)
	width = float(box[3]-box[1]+1)
	area = height*width
	
	if area == 0 or (area/im_area) > 0.8:
		return True
	
	if width < 20.0 or height < 20.0:
		return True
		
	ratio = width/height
	
	if ratio < 0.7 or ratio > 1.3:
		return True
	
	return False #don't filter out

#same for array of boxes
def filter_boxes(boxes, im_area):
	if len(boxes)==0:
		return np.array([])
	
	height = (boxes[:,2]-boxes[:,0]).astype(float) + 1.0
	width = (boxes[:,3]-boxes[:,1]).astype(float)+1.0
	rel_area = height*width/float(im_area)
	ratio = width/height
	
	filtered = np.where(rel_area < 0.8 and rel_area > 0 and width > 20.0 and height > 20.0 and ratio <= 1.3 and ratio >= 0.7)
	
	return boxes[filtered]
