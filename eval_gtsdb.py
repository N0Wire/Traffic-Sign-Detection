import numpy as np
import pandas as pd
import torch
from skimage import io
from torch.utils.data import DataLoader

#own stuff
from Classifier.dataloader import image, preprocessor, Evalset
from Classifier.trainer import import_classifier
from Classifier.network_utils import evaluate, visualize_stn

###########################################
"""
authors: Oliver Drodzdowski, Kim-Louis Simmoteit

This file loads resulting GT data from the GTSDB dataset
and let the network classify them.
"""
###########################################

path_to_boxes = "Detection/Runs/11/" 		#path to folder which contains files with bounding boxes with best overlaps
path_to_infos = "Data/FullIJCNN2013/"		#path to ground-truth data

if __name__ == "__main__":
	#load GT data
	info_table = pd.read_csv(path_to_infos + "gt.txt", delimiter=";", header=None)
	infos = np.array(info_table)

	Images = [] #list of image object
	preprocessor = preprocessor() # Load Preprocessor for the images
	#load boxes
	for i in range(len(infos)):
		im_name = infos[i][0]
		img = io.imread(path_to_infos+im_name)              
		temp = image(path_to_infos+im_name, infos[i][5], preprocessor,  img[infos[i][2]:infos[i][4]+1,infos[i][1]:infos[i][3]+1])
		Images.append(temp)

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
    
	# Visualize the STN on sample images of SVM output
	if use_gpu:
		model.databatch=next(iter(dataloader))["tensor"].cuda()
	else:
		model.databatch=next(iter(dataloader))["tensor"].cuda()
    
	visualize_stn(model, filename="./Classifier/Plots/stn_gtsdb_output.pdf")    
    
	print("For the GTSDB dataset (full) we obtain a total accuracy of the STN+CNN classifier of : " + str(accuracy))
