import numpy as np
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
	resized = transform.resize(cropped, (SIZE_X, SIZE_Y), anti_aliasing=True, mode="constant")
	
	desc = feature.hog(resized, pixels_per_cell=(6,6), cells_per_block=(2,2), visualize=False, block_norm="L1", transform_sqrt=True)
	return desc

#load bbox data
info_table = pd.read_csv(path + "gt.txt", delimiter=";", header=None)
infos = np.array(info_table)

#loads all data from first 600 runs and trains SVM
path2 = "Runs/"

def preprocess(start, num):
	for run_num in range(start, start+num):
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
			im_area = img.shape[0]*img.shape[1]
			
			rectangles = []
			#get GT-bboxes
			for e in infos:
				if e[0] == im_name:
					temp = [e[2], e[1], e[4], e[3]]
					rectangles.append(temp)
			
			#box, which has less than 50% IoU with GT-Data, is marked as negatives
			for b in box:
				#check if box fulfills rough sign criterium (70%<aspect-ratio<130% & min_width/height=20, area/im_area < 80)
				height = float(b[2]-b[0])
				width = float(b[3]-b[1])
				area = height*width
				
				if area == 0 or (area/im_area) > 0.8:
					continue
				
				ratio = width/height
				
				if ratio < 0.7 or ratio > 1.3:
					continue
				if width < 25.0 or height < 25.0:
					continue
				
				#box passed criterias
			
				use = True	#can bounding box be used as negative
				for k,r in enumerate(rectangles):
					score = overlap(r, b)
					cl = 0
					
					if score > 0.5: #no negative sample anymore
						use = False
						if score > 0.8:
							use = True
							cl = 1
				
				if use:
					crop = img[b[0]:b[2]+1,b[1]:b[3]+1]
					desc = GetDescriptor(crop)
					temp = [cl]
					for k in desc:
						temp.append(k)
					data.append(temp)
			
			#take GT-Data and and best IoU as positives if IoU is better than 50%
			for r in rectangles:
				crop = img[r[0]:r[2]+1,r[1]:r[3]+1]
				desc = GetDescriptor(crop)
				temp = [1]
				for k in desc:
					temp.append(k)
				data.append(temp)
		
		#save data
		ds = np.array(data)
		np.save("Data/hog_run_"+str(run_num)+".npy", ds, allow_pickle=False)


#maximum run=60 -> use first 600 images as training images
#last run: 1
preprocess(1,1)
