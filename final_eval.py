import numpy as np
import pandas as pd
import torch
from skimage import io
from torch.utils.data import DataLoader

#own stuff
from Classifier.dataloader import image, preprocessor, Evalset
from Classifier.trainer import import_classifier
from Classifier.network_utils import evaluate

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

	Images = [] #list of image object
	preprocessor = preprocessor() # Load Preprocessor for the images
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
						 temp = image(path_to_infos+im_name, e[5], preprocessor,  img[box[0]:box[2]+1,box[1]:box[3]+1])
						 Images.append(temp)
						#io.imsave("Test/"+str(run_num)+"_"+str(box_index)+".png", img[box[0]:box[2]+1,box[1]:box[3]+1]) #test output
					box_index += 1

	#Load our pretrained CNN-Model
	model, logger,_ = import_classifier()
    
	# Construct dataset and dataloader with Images
	evaluationset = Evalset(Images)
	dataloader = DataLoader(evaluationset, batch_size=32, shuffle=True)
    
	use_gpu = torch.cuda.is_available()
	if not use_gpu:
		print("Warning! No CUDA available. This is untested!")
    
	if use_gpu:
		model.cuda()
    
	# Calculate the accuracy of the pretrained model on the data from the SVM
	accuracy = evaluate(model, dataloader, use_gpu=use_gpu)
    
	print("For the output of the SVM we obtain a total accuracy of the STN+CNN classifier of : " + str(accuracy))
