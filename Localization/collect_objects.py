import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import time
import pickle
import multiprocessing as mp
import os

#own stuff
import sys
sys.path.insert(0,"../Detection")

import tools
from selective_search import SelectiveSearch


###########################################
"""
author: Kim-Louis Simmoteit

This script is used to obtain bounding boxes for all images
of the GTSDB dataset.
Those can later be used to get training data for the SVM or test 
the SVM without running Selective-Search again (saves time!).
"""
###########################################

#create directories if they don't exist
savepath_img = "./Images/"
savepath_run = "./Runs/"

if not os.path.exists(savepath_img):
	os.makedirs(savepath_img)
if not os.path.exists(savepath_run):
	os.makedirs(savepath_run)

#Setting Variables
plot = True 	#plot images with GT bounding box and best overlap


#Parse Database informations
path = "../Data/FullIJCNN2013/"		#path to GTSDB dataset
info_table = pd.read_csv(path + "gt.txt", delimiter=";", header=None)
infos = np.array(info_table)


#parse file
def collect(s_index, num_images, run_num):
	found_boxes = []		 	#list of all bounding boxes found
	best_overlaps = []			#contains best overlap score(IoU) for bounding box
	times = [] 					#contains time needed for selective search on every image

	start_index = s_index		#start image
	num = num_images 			#number of images

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
		
		#print("Evaluating " + im_name)
		#run selective search and obtain bounding boxes
		img = plt.imread(im_path)
		
		start = time.time()
		ss = SelectiveSearch()
		boxes = ss.run(img, "deep") #one could use other methods here -> deep is a good trade off between speed and accuracy
		stop = time.time()
		times.append(stop-start)
		found_boxes.append(boxes) 
		
		#determine best overlapp
		for i,r in enumerate(rectangles):
			for b in boxes:
				score = tools.overlap(r, b)
				if score > scores[i]:
					scores[i] = score
					overlap_boxes[i] = b
		
		best_overlaps += scores
		
		#plotting images with best overlaps
		if plot:
			plt.figure(1, dpi=100, figsize=(13.6,8.0))
			plt.clf()
			plt.axis("off")
			plt.imshow(img)
			
			for r in rectangles: #ground truth data
				rect = patches.Rectangle((r[1], r[0]),np.abs(r[3]-r[1]), np.abs(r[2]-r[0]), linewidth=1, edgecolor="g", facecolor="none")
				plt.gca().add_patch(rect)
			for b in overlap_boxes: #best matching bounding boxes
				rect = patches.Rectangle((b[1], b[0]),np.abs(b[3]-b[1]), np.abs(b[2]-b[0]), linewidth=1, edgecolor="r", facecolor="none")
				plt.gca().add_patch(rect)
			plt.savefig(savepath_img + im_name[0:len(im_name)-3] + "png") #can't save as ppm -> save as png

	#reduce data
	bos = np.array(best_overlaps)
	ts = np.array(times)

	#save data for later use
	np.save(savepath_run+"overlap_"+str(run_num)+".npy", bos)
	np.save(savepath_run+"time_"+str(run_num)+".npy", ts)
	f = open(savepath_run+"boxes_"+str(run_num)+".dat", "wb")
	pickle.dump(found_boxes,f)
	f.close()


#run script with only one of them uncommented !
#-> would take too much time to take data in one go!

#per "batch" 50 images
#training images
start_values = [[0, 10, 1],[10, 10, 2],[20, 10, 3],[30, 10, 4],[40, 10, 5]]					#1
#start_values = [[50,10, 6],[60, 10, 7],[70, 10, 8],[80, 10, 9],[90, 10, 10]]					#2
#start_values = [[100, 10, 11],[110, 10, 12],[120, 10, 13],[130, 10, 14],[140, 10, 15]]			#3
#start_values = [[150,10, 16],[160, 10, 17],[170, 10, 18],[180, 10, 19],[190, 10, 20]]			#4
#start_values = [[200, 10, 21],[210, 10, 22],[220, 10, 23],[230, 10, 24],[240, 10, 25]]			#5
#start_values = [[250, 10, 26],[260, 10, 27],[270, 10, 28],[280, 10, 29],[290, 10, 30]]			#6
#start_values = [[300, 10, 31],[310, 10, 32],[320, 10, 33],[330, 10, 34],[340, 10, 35]]			#7
#start_values = [[350, 10, 36],[360, 10, 37],[370, 10, 38],[380, 10, 39],[390, 10, 40]]			#8
#start_values = [[400, 10, 41],[410, 10, 42],[420, 10, 43],[430, 10, 44],[440, 10, 45]]			#9
#start_values = [[450, 10, 46],[460, 10, 47],[470, 10, 48],[480, 10, 49],[490, 10, 50]]			#10
#start_values = [[500, 10, 51],[510, 10, 52],[520, 10, 53],[530, 10, 54],[540, 10, 55]]			#11
#start_values = [[550, 10, 56],[560, 10, 57],[570, 10, 58],[580, 10, 59],[590, 10, 60]]			#12

#test samples
#start_values = [[600, 10, 61],[610, 10, 62],[620, 10, 63],[630, 10, 64],[640, 10, 65]]			#13
#start_values = [[650, 10, 66],[660, 10, 67],[670, 10, 68],[680, 10, 69],[690, 10, 70]]			#14
#start_values = [[700, 10, 71],[710, 10, 72],[720, 10, 73],[730, 10, 74],[740, 10, 75]]			#15
#start_values = [[750, 10, 76],[760, 10, 77],[770, 10, 78],[780, 10, 79],[790, 10, 80]]			#16
#start_values = [[800, 10, 81],[810, 10, 82],[820, 10, 83],[830, 10, 84],[840, 10, 85]]			#17
#start_values = [[850, 10, 86],[860, 10, 87],[870, 10, 88],[880, 10, 89],[890, 10, 90]]			#18

#Use multithreading
procs = []
for v in start_values:
	p = mp.Process(target=collect, args=(v[0], v[1], v[2]))
	procs.append(p)

for i,p in enumerate(procs):
	print("Start " + str(start_values[i][2]))
	p.start()

for p in procs:
	p.join()

