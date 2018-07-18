import numpy as np
import matplotlib.pyplot as plt
from skimage import io

#own stuff
from Localization.selective_search import SelectiveSearch
from Detection.sign_checker import SignChecker
from Classifier.dataloader import image
from Classifier.signname import gtsrb_signname
from Classifier.trainer import import_classifier
from Classifier.network_utils import predict

###########################################
"""
authors: Oliver Drodzdowski, Kim-Louis Simmoteit

This file loads images and runs our traffic sign
detector over them.
"""
###########################################

def find_traffic_signs(img, visualize=False):
	"""
	This function applies the entire algorithm on an image represented as a numpy array (e.g. from imread)
	First bounding boxes for region proposals are calculated via the Selective Search Algorithm
	All these bounding boxes are fed into the SVM to determine, which bounding boxes actually contain
	a traffic sign.
	After this the Conv.Neur.Net. with Spat.Transf.Net. is used to classify the following bounding
    boxes.
	Lastly, the image is visualized with bounding boxes if wanted
	Arguments:     img - image as numpy array
                    visualize - (boolean) whether to plot and save the image with bounding boxes
	Return:    bounding_list - list of bounding boxes in format []
                ClassIds - list of corresponding classids
                TrafficSignNames - ist of corresponding names of signs
	"""
    #load stuff
	s_search = SelectiveSearch()	#selective search
	checker = SignChecker()			#SVM
	checker.load("Detection/SVMs/11_")
	
    # Create preprocessor and import pretrained model
	preprocessor = preprocessor()
	model,_,_ = import_classifier()
    
    
	proposals = s_search.run(img, method="deep")
	signs = checker.check(img, proposals)
	
	# Create list of image objects, which are needed for the CNN_STN classifier    
	Images = [] #list of image objects
	for s in signs:
		im = image("", -1, preprocessor, img[s[0]:s[2]+1,s[1]:s[3]+1])
		Images.append(im)
	
	# Feed all the image objects into the classifier to obtain the classids and names
	ClassIds = []
	TrafficSignNames = []
    
	for image in Images:
		# Predict class id 
		prediction = predict(model, image)
		ClassIds.append(prediction)
        
		# Look up name corresponding to classid
		signname = gtsrb_signname(prediction)
		TrafficSignNames.append(signname)
        
	bounding_list = signs
	        
	return bounding_list, ClassIds, TrafficSignNames
    
if __name__ == "__main__":
	#load images
	
	#do stuff
