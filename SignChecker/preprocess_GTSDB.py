import numpy as np
import pandas as pd
import pickle
from skimage import io
import multiprocessing as mp
import glob
import tools

#use resulting boxes from Selective Search to get negative training samples

#Setting Variables
path = "../Data/FullIJCNN2013/"

#load bbox data
info_table = pd.read_csv(path + "gt.txt", delimiter=";", header=None)
infos = np.array(info_table)

#loads all data from first 600 runs and trains SVM
path2 = "Runs/"

#image to take negative samples from (try to take different locations)
negative_samples = [4, 35, 53, 57, 113, 126, 210, 418, 465, 544, 41, 143, 236, 318, 367, 460, 439, 555, 585, 595, 64, 70, 103]

#maximum run=60 -> use first 600 images as training images
def process_run(run_num, negatives=[]):
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
			#check if box fullfills rough sign criterium
			if tools.filter_box(b, im_area):
				continue
			
			#box passed criterias
			use = False	#can bounding box be used as negative
			cl = 0
			for k,r in enumerate(rectangles):
				score = tools.overlap(r, b)
				
				if score < 0.5 and im_num in negatives: #no negative sample anymore
					use = True
				elif score > 0.8:
					use = True
					cl = 1
			
			if use:
				crop = img[b[0]:b[2]+1,b[1]:b[3]+1]
				desc = tools.HogDescriptor(crop)
				temp = [cl]
				for k in desc:
					temp.append(k)
				data.append(temp)
		
		#take GT-Data as positives
		for r in rectangles:
			crop = img[r[0]:r[2]+1,r[1]:r[3]+1]
			desc = tools.HogDescriptor(crop)
			temp = [1]
			for k in desc:
				temp.append(k)
			data.append(temp)
	
	#save data
	ds = np.array(data)
	np.save("Data/hog_run_"+str(run_num)+".npy", ds, allow_pickle=False)


#maximum run=60 -> use first 600 images as training images
index = 1
while index <= 56:
	procs = []
	for v in range(index, index+5):
		p = mp.Process(target=process_run, args=(v, negative_samples))
		procs.append(p)

	for i,p in enumerate(procs):
		p.start()

	for p in procs:
		p.join()
	
	index += 5

#additional signs from GTSDB
print("Processing GTSDB signs")
descriptors = []
for i in range(43): #43 classes are existing
	full_path = "{}{:02d}/".format(path, i)
	files = glob.glob(full_path+"*.ppm")
	for f in files:
		img = io.imread(f)
		desc = tools.HogDescriptor(img)
		temp = [1]
		for k in desc:
			temp.append(k)
		descriptors.append(temp)

ds = np.array(descriptors)
np.save("Data/gtsdb.npy", ds, allow_pickle=False)
