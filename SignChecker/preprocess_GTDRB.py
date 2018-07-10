import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from skimage import io, transform, feature

#use resulting boxes from Selective Search to get negative training samples

#Setting Variables
path = "../Data/FullIJCNN2013/"
SIZE_X = 64		#size of final image in x direction
SIZE_Y = 64		#size of final image in y direction

#Intersection over Union
def overlap(box1, box2):
	area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
	area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
	
	#intersection region
	miny = max(box1[0], box2[0])
	minx = max(box1[1], box2[1])
	maxy = min(box1[2], box2[2])
	maxx = min(box1[3], box2[3])
	
	intersect = max(0, maxy-miny)*max(0, maxx-minx)
	total_area = float(area1+area2-intersect) #don't count area 2 times
	
	return intersect/total_area

def GetDescriptor(cropped):
	resized = transform.resize(cropped, (SIZE_X, SIZE_Y), anti_aliasing=True)
	
	desc = feature.hog(resized, pixels_per_cell=(6,6), cells_per_block=(2,2), visualize=False, block_norm="L1")
	return desc

#load bbox data
info_table = pd.read_csv(path + "gt.txt", delimiter=";", header=None)
infos = np.array(info_table)

#loads all data from first 600 runs and trains SVM
path2 = "Runs/"
boxes = []

max_runnum = 2 #collect first 600 images as training images
for run_num in range(1, max_runnum+1):
	print("Processing Run " + str(run_num))
	f = open(path2+"boxes_"+str(run_num)+".dat", "rb")
	bs = pickle.load(f)
	f.close()
	
	#for each run collect data
	data = []
	#crop out image parts
	for i,box in enumerate(bs):
		#load image
		im_num = 10*run_num+i #each run contains 10 images
		im_name = "{:05d}.ppm".format(im_num)
		img = io.imread(path+im_name)
		
		rectangles = []
		overlap_boxes = []
		scores = []
		#get GT-bboxes
		for e in infos:
			if e[0] == im_name:
				temp = [e[2], e[1], e[4], e[3]]
				rectangles.append(temp)
				overlap_boxes.append([0,0,0,0]) #dummy bounding box
				scores.append(0)
		
		#box, which has less than 50% IoU with GT-Data, is marked as negatives
		for b in box:
			use = True	#can bounding box be used as negative
			for k,r in enumerate(rectangles):
				score = overlap(r, b)
				
				if score > 0.5: #no negative sample anymore
					use = False
				if score > scores[k]:
					scores[k] = score
					overlap_boxes[k] = b
			
			crop = img[b[0]:b[2]+1,b[1]:b[3]+1]
			desc = GetDescriptor(crop)
			temp = [0, desc]
			data.append(temp)
		
		#take GT-Data and and best IoU as positives if IoU is better than 50%
		for r in rectangles:
			crop = img[r[0]:r[2]+1,r[1]:r[3]+1]
			desc = GetDescriptor(crop)
			temp = [1, desc]
			data.append(temp)
		
		for k,b in enumerate(overlap_boxes):
			if scores[k] < 0.5:
				continue
			crop = img[b[0]:b[2]+1,b[1]:b[3]+1]
			desc = GetDescriptor(crop)
			temp = [1, desc]
			data.append(temp)
	
	#save data
	ds = np.array(data)
	np.save("Data/hog_run_"+str(i)+".npy", ds)
