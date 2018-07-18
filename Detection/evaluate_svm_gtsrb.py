import numpy as np
import pandas as pd
from skimage import io

#own stuff
from sign_checker import SignChecker
import tools

###########################################
"""
author: Kim-Louis Simmoteit

This script is used to check the SVM
with the test data of the GTSRB dataset.
"""
###########################################

#load SVM
sc = SignChecker()
sc.load("SVMs/4_")

path = "../Data/GTSRB/Final_Test/Images/"

#load infos
info_table = pd.read_csv(path+"GT-final_test.test.csv", delimiter=";")
infos = np.array(info_table)

pred = np.zeros(len(infos))
for i in range(len(infos)):
	full_path = path + infos[i][0]
	img = io.imread(full_path)
	desc = tools.HogDescriptor(img)
	pred[i] = sc.check_desc([desc])[0]

classification = np.sum(pred)/len(infos)
print("Accuarcy on GTSRB dataset: " + str(classification))
