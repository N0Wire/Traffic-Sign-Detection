import numpy as np
import pandas as pd
from skimage import io

#own stuff
from Classifier.dataloader import image

###########################################
"""
authors: Oliver Drodzdowski, Kim-Louis Simmoteit

This file loads resulting bounding boxes (best overlaps after SVM) 
from the 300 test images
and let the network classify them.
"""
###########################################

path_to_boxes = "Detection/Runs/11/" 		#path to folder which contains files with bounding boxes with best overlaps
path_to_infos = "Data/FullIJCNN2013/"		#path to ground-truth data

if __name__ == "__main__":
	#load GT data
	info_table = pd.read_csv(path_to_infos + "gt.txt", delimiter=";", header=None)
	infos = np.array(info_table)[852:,:] #gt data needed at i=853-1s

	Images = [] #array of image object
	#load boxes
	max_runnum = 90
	for run_num in range(61, max_runnum+1):
		#overlaps = np.load(path_to_boxes+"overlaps_"+str(run_num)+".npy")
		obs = np.load(path_to_boxes+"overlapboxes_"+str(run_num)+".npy")
		
		box_index = 0
		for i in range(0, 10):
			im_name = "{:05d}.ppm".format((run_num-1)*10+i) #run_numbers start at 1 [1->image 0(+10), 61->image 600(+10)]
			img = io.imread(path_to_infos+im_name)
			for e in infos:
				if e[0] == im_name:
					if not np.array_equal(obs[box_index],[0,0,0,0]): #best overlap exists
						box = obs[box_index]
						temp = image(path_to_infos+im_name, e[5], None,  img[box[0]:box[2]+1,box[1]:box[3]+1])
						Images.append(temp)
						#io.imsave("Test/"+str(run_num)+"_"+str(box_index)+".png", img[box[0]:box[2]+1,box[1]:box[3]+1]) #test output
					box_index += 1

	#Load CNN-Model

	#feed CNN with data
	
	#classify
