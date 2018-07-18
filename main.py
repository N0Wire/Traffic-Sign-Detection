import numpy as np
import matplotlib.pyplot as plt
from skimage import io

#own stuff
from Localization.selective_search import SelectiveSearch
from Detection.sign_checker import SignChecker
from Classifier.dataloader import image

###########################################
"""
authors: Oliver Drodzdowski, Kim-Louis Simmoteit

This file loads images and runs our traffic sign
detector over them.
"""
###########################################

def find_traffic_signs(img, visualize=False):
	#load stuff
	s_search = SelectiveSearch()	#selective search
	checker = SignChecker()			#SVM
	checker.load("Detection/SVMs/11_")
	#Network here
	
	proposals = s_search.run(img, method="deep")
	signs = checker.check(img, proposals)
	
	Images = [] #list of image objects
	for s in signs:
		im = image("", -1, img[s[0]:s[2]+1,s[1]:s[3]+1], canny)
		Images.append(im)
	
	#put images in classifier

if __name__ == "__main__":
	#load images
	
	#do stuff