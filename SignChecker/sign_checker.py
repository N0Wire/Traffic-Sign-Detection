import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
import tools

#class which extracts only bounding boxes which contain traffic signs
class SignChecker:
	
	def __init__(self, cval=1.0, gam="auto"):
		#Support Vector Machine
		self.svm = SVC(kernel="rbf", C=cval, gamma=gam, cache_size=1000)
	
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
			c = self.svm.predict([desc])
			
			#if is sign
			if c == 1.0:
				signs.append(b)
		
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
		
		return res

