import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from skimage import io, transform, feature
from sign_checker import SignChecker

#load bbox data
path = "../Data/FullIJCNN2013/"
infos = pd.read_csv(path + "gt.txt", delimiter=";", header=None)

#loads all data from first 600 runs and and creates histograms and trains SVM
path2 = "Runs/"
best_overlapps = []
times = []
boxes = []

max_runnum = 60
for run_num in range(1, max_runnum+1):
	f = open(path2+"boxes_"+str(run_num)+".dat", "rb")
	bs = pickle.load(f)
	f.close()

	boxes.append(bs)



#crop out image parts
#for i,b in enumerate(boxes):
#	



#sc = SignChecker(1.0)
#sc.train()
#sc.save()
