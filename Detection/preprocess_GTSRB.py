import numpy as np
from skimage import io
import pandas as pd
import multiprocessing as mp
import os

#own stuff
import tools

#create directories if they don't exist
output_path = "./Train_data/"
if not os.path.exists(output_path):
	os.makedirs(output_path)

#Take images from GTSRB database and calculate HOG-Descriptors
#-> those are later used to train SVM

#Setting Variables
num_classes = 43 #(0-42)
path_training = "../Data/GTSRB/Final_Training/Images"

#use different parts of image, scales them and calculates HOG-Descriptors
#path: path to data-set folder (with / at end)
#name: name of file
#sign_class: class number of sign
def evaluate_image(path, name, bbox):
	descs = []
	full_path = path + name
	
	img = io.imread(full_path)
	
	#full image
	desc = tools.HogDescriptor(img)
	descs.append(desc)
	
	#bounding box
	desc = tools.HogDescriptor(img[bbox[0]:bbox[2]+1,bbox[1]:bbox[3]+1])
	descs.append(desc)
	
	#cut parts of the sign (always substract about 5 pixels)
	"""
	#top
	desc = tools.HogDescriptor(img[bbox[0]+5:,:])
	descs.append(desc)
	
	#bottom
	desc = tools.HogDescriptor(img[:bbox[2]+1-5,:])
	descs.append(desc)
	
	#left
	desc = tools.HogDescriptor(img[:,bbox[1]+5:])
	descs.append(desc)
	
	#right
	desc = tools.HogDescriptor(img[:,:bbox[3]+1-5])
	descs.append(desc)
	"""
	#center (only crop 2 pixels)
	desc = tools.HogDescriptor(img[bbox[0]+2:bbox[2]+1-2,bbox[1]+2:bbox[3]+1-2])
	descs.append(desc)
	
	
	return descs

####################
#Training Images
print("[+]Training Images")


#preprocess
def process_class(c_index):
	print("Processing Image-Class " + str(c_index))
	full_path = "{}/{:05d}/".format(path_training, c_index)
	csvfile = "{}GT-{:05d}.csv".format(full_path, c_index)
	
	#load data
	data = pd.read_csv(csvfile, delimiter=";")
	names = data["Filename"]
	
	x1 = data["Roi.X1"]
	x2 = data["Roi.X2"]
	y1 = data["Roi.Y1"]
	y2 = data["Roi.Y2"]
	
	#load pictures and preprocess
	descriptors = []
	j = 0
	while j < len(names):
		#if j > 1500:	#maximum 50 tracks (30)
		#	break
		box = [y1[j], x1[j], y2[j], x2[j]]
		descs = evaluate_image(full_path, names[j], box)
		for d in descs:
			temp = [1] #1 for Traffic Sign - 0 for no traffic sign
			for k in d:
				temp.append(k)
			descriptors.append(temp)
		#each track contains 30 images -> take every fifth -> 6 images per track
		j += 3
	
	#save data
	ds = np.array(descriptors)
	np.save(output_path+"hog_train_c"+str(c_index)+".npy", ds, allow_pickle=False)


##########################################################
start_values = range(num_classes)

procs = []
for v in start_values:
	p = mp.Process(target=process_class, args=(v,))
	procs.append(p)

for i,p in enumerate(procs):
	print("Start " + str(start_values[i]))
	p.start()

for p in procs:
	p.join()
