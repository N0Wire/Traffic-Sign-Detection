import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib

#oen stuff
import tools

###########################################
"""
author: Kim-Louis Simmoteit

This module contains the class for detecting traffic
signs with a SVM.
"""
###########################################

#class which extracts only bounding boxes which contain traffic signs
class SignChecker:
	
	def __init__(self, cval=1.0, gam="auto"):
		#Support Vector Machine
		self.svm = SVC(kernel="rbf", C=cval, gamma=gam, cache_size=2000)
	
	#save SVM to file
	#path = path with / at end
	def save(self, path):
		joblib.dump(self.svm, path + "svm.sav")
	
	#load SVM from file
	#path = path with / at end
	def load(self, path):
		self.svm = joblib.load(path + "svm.sav")
	
	
	def train(self, hog_desc, labels):
		self.svm.fit(hog_desc, labels)
	
	# Malisiewicz et al. (https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/)
	def NMS(self, boxes, thr):
		# if there are no boxes, return an empty list
		if len(boxes) == 0:
			return []
	 
		# if the bounding boxes integers, convert them to floats --
		# this is important since we'll be doing a bunch of divisions
		if boxes.dtype.kind == "i":
			boxes = boxes.astype("float")
	 
		# initialize the list of picked indexes	
		pick = []
	 
		# grab the coordinates of the bounding boxes
		y1 = boxes[:,0]
		x1 = boxes[:,1]
		y2 = boxes[:,2]
		x2 = boxes[:,3]
	 
		# compute the area of the bounding boxes and sort the bounding
		# boxes by the bottom-right y-coordinate of the bounding box
		area = (x2 - x1 + 1) * (y2 - y1 + 1)
		idxs = np.argsort(y2)
	 
		# keep looping while some indexes still remain in the indexes
		# list
		while len(idxs) > 0:
			# grab the last index in the indexes list and add the
			# index value to the list of picked indexes
			last = len(idxs) - 1
			i = idxs[last]
			pick.append(i)
	 
			# find the largest (x, y) coordinates for the start of
			# the bounding box and the smallest (x, y) coordinates
			# for the end of the bounding box
			xx1 = np.maximum(x1[i], x1[idxs[:last]])
			yy1 = np.maximum(y1[i], y1[idxs[:last]])
			xx2 = np.minimum(x2[i], x2[idxs[:last]])
			yy2 = np.minimum(y2[i], y2[idxs[:last]])
	 
			# compute the width and height of the bounding box
			w = np.maximum(0, xx2 - xx1 + 1)
			h = np.maximum(0, yy2 - yy1 + 1)
	 
			# compute the ratio of overlap
			overlap = (w * h) / area[idxs[:last]]
	 
			# delete all indexes from the index list that have
			idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > thr)[0])))
	 
		# return only the bounding boxes that were picked using the
		# integer data type
		return boxes[pick].astype("int")
	
	def check_desc(self, desc):
		return self.svm.predict(desc)
	
	#check a set of bounding boxes for signs (not tested yet)
	def check(self, img, boxes):
		im_area = float(img.shape[0]*img.shape[1])
		#boxes = np.array(boxes)
		#first filter boxes by aspect ratio
		filtered_boxes = []
		for b in boxes:
			if tools.filter_box(b, im_area):
				continue
			filtered_boxes.append(b)
		
		signs = []
		
		#now calculate HOG-Descriptor for every Box
		for b in filtered_boxes:
			desc = tools.HogDescriptor(img[b[0]:b[2]+1,b[1]:b[3]+1])
			c = self.svm.predict([desc])[0]
			
			#if is sign
			if c == 1.0:
				signs.append(b)
		
		#old non maximum suppression
		"""
		#now check if one has found the same sign multiple times (if IoU>=80% -> average boxes)
		res = []
		
		for i,s in enumerate(signs):
			indexes = []
			bbs = []
			tl = []
			tr = []
			bl = []
			br = []
			for j, k in enumerate(signs):
				score = tools.overlap(s, k)
				if score >= 0.75:
					indexes.append(j)
					bbs.append(k)
					tl.append(k[0])
					tr.append(k[1])
					bl.append(k[2])
					br.append(k[3])
			
			#average and remove other bounding boxes
			new_b = [0,0,0,0]
			new_b[0] = int(np.mean(tl))
			new_b[1] = int(np.mean(tr))
			new_b[2] = int(np.mean(bl))
			new_b[3] = int(np.mean(br))
			
			for l in reversed(indexes):
				signs.pop(l)
			
			res.append(new_b)
		"""
		
		#non maximu suppression 60% threshold
		return self.NMS(np.array(signs), 0.6)

