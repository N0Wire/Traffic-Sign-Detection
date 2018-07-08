import numpy as np
from skimage import io, transform, feature
from sklearn.svm import SVC
from sklearn.externals import joblib

#class which extracts only bounding boxes which contain traffic signs
class SignChecker:
	
	def __init__(self, cval=1.0, gam="auto"):
		#Support Vector Machine
		self.svm = SVC(kernel="rbf", C=cval, gamma=gam)
	
	#save SVM to file
	#path = path with / at end
	def save(self, path):
		joblib.dump(self.svm, path + "svm.sav")
	
	#load SVM from file
	#path = path with / at end
	def load(self, path):
		self.svm = joblib.load(path + "svm.sav")
		
	
	#Intersection over Union
	def overlap(self, box1, box2):
		area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
		area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
		
		#intersection region
		miny = max(box1[0], box2[0])
		minx = max(box1[1], box2[1])
		maxy = min(box1[2], box2[2])
		maxx = min(box1[3], box2[3])
		
		intersect = max(0, maxy-miny)*max(0, maxx-minx)
		total_area = float(area1+area2-intersect) #don't count area 2 times
		
		return intersect/total_area
	
	#check a set ob bounding boxes for signs (not tested yet)
	def check(self, img, boxes):
		boxes = np.array(boxes)
		#first filter boxes by aspect ratio
		heights = boxes[:,2]-boxes[:,0]
		widths = boxes[:,3]-boxes[:,1]
		ratios = heights/widths
		collective = np.array((boxes.shape[0], 2))
		collective[:,0] = np.arange(boxes.shape[0])
		collective[:,1] = ratios
		
		sort = collective[collective[:,1].argsort()]
		
		indexes = np.where(sort[:,1]>= 0.7 and sort[:,1]<= 1.3) #+-30% deviation of a square
		
		filtered_boxes = sort[indexes]
		
		signs = []
		
		#now calculate HOG-Descriptor for every Box
		for b in filtered_boxes:
			cropped = img[b[0]:b[2]+1,b[1]:b[3]+1]
			scaled = transform.resize(cropped, (64,64))
			desc = feature.hog(scaled, pixels_per_cell=(6,6), cells_per_block=(2,2), visualize=False, block_norm="L1")
			c = self.svm.predict(desc)
			
			#if is sign
			if c==1:
				signs.append(b)
		
		#now check if one has found the same sign multiple times (use IoU and a threshold)
		
		
