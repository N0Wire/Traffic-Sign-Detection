import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import time
import pickle
import tools
import multiprocessing as mp
import os

#own stuff
from sign_checker import SignChecker

###########################################
"""
author: Kim-Louis Simmoteit

This script is used to load collected bounding boxes (selective search)
and push them into the SVM to get only bounding boxes which are signs.
"""
###########################################

#Settings
plot = False
svm_prefix = "11_"

#create directories if they don't exist
output_path = "./Runs/"+svm_prefix[0:len(svm_prefix)-1]+"/"
if not os.path.exists(output_path):
	os.makedirs(output_path)

output_path_im = "./Images/"
if plot:
	if not os.path.exists(output_path_im):
		os.makedirs(output_path_im)

svm_path = "./SVMs/"+svm_prefix


#parse file
path = "../Data/FullIJCNN2013/"
path_boxes = "../Localization/Runs/"		#load boxes received from selective search
info_table = pd.read_csv(path + "gt.txt", delimiter=";", header=None)
infos = np.array(info_table)


#collect signs from a given Run
def collect(s_index, num_images, run_num):
	best_overlaps = []			#contains best overlap score(IoU) for bounding box
	times = [] 					#contains time needed for selective search on every image
	overlap_bos = []			#best matching bounding boxes
	wrong_bos = [] 				#additional boxes which are marked as signs but don't belong to any GT data 
	
	#load run
	f = open(path_boxes+"boxes_"+str(run_num)+".dat", "rb")
	bs = pickle.load(f)
	f.close()
	

	for i,bboxes in enumerate(bs):
		im_name = "{:05d}.ppm".format(s_index+i)
		im_path = path+im_name
		
		rectangles = []
		overlap_boxes = []
		scores = []
		wrong_boxes = [] #additional boxes which are marked as signs but don't belong to any GT data 
		
		for e in infos:
			if e[0] == im_name:
				temp = [e[2], e[1], e[4], e[3]]
				rectangles.append(temp)
				overlap_boxes.append([0,0,0,0]) #dummy bounding box
				scores.append(0)
		
		#print("Evaluating " + im_name)
		#run SignChecker to get signs from all bounding boxes
		img = plt.imread(im_path)
		
		start = time.time()
		sc = SignChecker()
		sc.load(svm_path)
		boxes = sc.check(img, bboxes)
		stop = time.time()
		times.append(stop-start)
		
		#determine best overlapp
		for j,r in enumerate(rectangles):
			for b in boxes:
				score = tools.overlap(r, b)
				if score > scores[j]:
					scores[j] = score
					overlap_boxes[j] = b
		
		best_overlaps += scores
		
		#get wrong boxes
		for b in boxes:
			for o in overlap_boxes:
				if np.array_equal(b, o):
					continue
				wrong_boxes.append(b)
		
		if len(overlap_boxes)==0:
			wrong_boxes = boxes
		if not isinstance(wrong_boxes, list):	
			wrong_bos += wrong_boxes.tolist()
		else:
			wrong_bos += wrong_boxes
		overlap_bos += overlap_boxes
		
		if plot:
			plt.figure(1, dpi=100, figsize=(13.6, 8.0))
			plt.clf()
			plt.axis("off")
			plt.imshow(img)
			
			for r in rectangles: #given rectangles
				rect = patches.Rectangle((r[1], r[0]),np.abs(r[3]-r[1]), np.abs(r[2]-r[0]), linewidth=1, edgecolor="g", facecolor="none")
				plt.gca().add_patch(rect)
			for b in wrong_boxes:
				rect = patches.Rectangle((b[1], b[0]),np.abs(b[3]-b[1]), np.abs(b[2]-b[0]), linewidth=1, edgecolor="b", facecolor="none")
				plt.gca().add_patch(rect)
			for b in overlap_boxes: #best matching bounding boxes
				rect = patches.Rectangle((b[1], b[0]),np.abs(b[3]-b[1]), np.abs(b[2]-b[0]), linewidth=1, edgecolor="r", facecolor="none")
				plt.gca().add_patch(rect)
			plt.savefig(output_path_im + im_name[0:len(im_name)-3] + "png") #can't save as ppm -> save as png

	#reduce data
	bos = np.array(best_overlaps)
	ts = np.array(times)
	wbs = np.array(wrong_bos)
	obs = np.array(overlap_bos)

	#save data for later use
	np.save(output_path+"overlap_"+str(run_num)+".npy", bos, allow_pickle=False)
	np.save(output_path+"times_"+str(run_num)+".npy", ts, allow_pickle=False)
	np.save(output_path+"wrongboxes_"+str(run_num)+".npy", wbs, allow_pickle=False)
	np.save(output_path+"overlapboxes_"+str(run_num)+".npy", obs, allow_pickle=False)


#run script with only one of them uncommented !
#-> would take too much time to take data in one go!

#per "batch" 50 images
#test samples
start_values = [[600, 10, 61],[610, 10, 62],[620, 10, 63],[630, 10, 64],[640, 10, 65]]			#13
#start_values = [[650, 10, 66],[660, 10, 67],[670, 10, 68],[680, 10, 69],[690, 10, 70]]			#14
#start_values = [[700, 10, 71],[710, 10, 72],[720, 10, 73],[730, 10, 74],[740, 10, 75]]			#15
#start_values = [[750, 10, 76],[760, 10, 77],[770, 10, 78],[780, 10, 79],[790, 10, 80]]			#16
#start_values = [[800, 10, 81],[810, 10, 82],[820, 10, 83],[830, 10, 84],[840, 10, 85]]			#17
#start_values = [[850, 10, 86],[860, 10, 87],[870, 10, 88],[880, 10, 89],[890, 10, 90]]			#18

#output = mp.Queue()
procs = []
for v in start_values:
	p = mp.Process(target=collect, args=(v[0], v[1], v[2]))
	procs.append(p)

for i,p in enumerate(procs):
	print("Start " + str(start_values[i][2]))
	p.start()

for p in procs:
	p.join()
