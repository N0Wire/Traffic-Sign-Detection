import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from selective_search import SelectiveSearch
import time


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

#parse file
path = "../Data/FullIJCNN2013/"
savepath = "Localization/"
info_table = pd.read_csv(path + "gt.txt", delimiter=";", header=None)

infos = np.array(info_table)

found_boxes = []		 	#list of all bounding boxes found
best_overlaps = []			#contains best overlap score(IoU) for bounding box
times = [] 					#contains time needed for selective search on every image

start_index = 0				#start image
num = 3 					#number of images

for i in range(start_index, start_index+num):
	im_name = "{:05d}.ppm".format(i)
	im_path = path+im_name
	
	rectangles = []
	overlap_boxes = []
	scores = [] 
	
	for e in infos:
		if e[0] == im_name:
			temp = [e[2], e[1], e[4], e[3]]
			rectangles.append(temp)
			overlap_boxes.append([0,0,0,0]) #dummy bounding box
			scores.append(0)
	
	print("Evaluating " + im_name)
	#run selective search and obtain bounding boxes
	img = plt.imread(im_path)
	
	start = time.time()
	ss = SelectiveSearch()
	boxes = ss.run(img, "deep") # "fast"
	stop = time.time()
	times.append(stop-start)
	found_boxes.append(boxes) 
	
	#determine best overlapp
	for i,r in enumerate(rectangles):
		for b in boxes:
			score = overlap(r, b)
			if score > scores[i]:
				scores[i] = score
				overlap_boxes[i] = b
	
	best_overlaps += scores
	
	plt.figure(1, dpi=100, figsize=(13.6,8.0))
	plt.clf()
	plt.axis("off")
	plt.imshow(img)
	
	for r in rectangles: #given rectangles
		rect = patches.Rectangle((r[1], r[0]),np.abs(r[3]-r[1]), np.abs(r[2]-r[0]), linewidth=1, edgecolor="g", facecolor="none")
		plt.gca().add_patch(rect)
	for b in overlap_boxes: #best matching bounding boxes
		rect = patches.Rectangle((b[1], b[0]),np.abs(b[3]-b[1]), np.abs(b[2]-b[0]), linewidth=1, edgecolor="r", facecolor="none")
		plt.gca().add_patch(rect)
	plt.savefig(savepath + im_name[0:len(im_name)-3] + "png") #can't save as ppm -> save as png



#reduce data
best_overlaps = np.array(best_overlaps)
times = np.array(times)
print(best_overlaps)

#calculate mean best overlap:
mbo = np.sum(best_overlaps)/float(best_overlaps.shape[0])
print("Mean Best Overlapp: " + str(mbo))

mt = np.sum(times)/float(times.shape[0])
print("Mean Time needed for selective search: " + str(mt))
