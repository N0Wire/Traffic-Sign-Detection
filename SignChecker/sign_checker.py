import numpy as np
from skimage import io, transform, feature
from sklearn.svm import SVC
from sklearn.externals import joblib

SIZE_X = 64		#size of final image in x direction
SIZE_Y = 64		#size of final image in y direction

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
	
	def check_desc(self, desc):
		return self.svm.predict(desc)
	
	#check a set of bounding boxes for signs (not tested yet)
	def check(self, img, boxes):
		im_area = float(img.shape[0]*img.shape[1])
		#boxes = np.array(boxes)
		#first filter boxes by aspect ratio
		"""
		heights = boxes[:,2]-boxes[:,0]
		widths = boxes[:,3]-boxes[:,1]
		ratios = heights/widths
		collective = np.array((boxes.shape[0], 2))
		collective[:,0] = np.arange(boxes.shape[0])
		collective[:,1] = ratios
		
		sort = collective[collective[:,1].argsort()]
		
		indexes = np.where(sort[:,1]>= 0.7 and sort[:,1]<= 1.3) #+-30% deviation of a square
		
		filtered_boxes = sort[indexes]
		"""
		filtered_boxes = []
		for b in boxes:
			height = float(b[2]-b[0])
			width = float(b[3]-b[1])
			area = height * width
			if width < 20.0 or height < 20.0:	#too small
				continue
			if area == 0 or (area/im_area) > 0.8: #too big
				continue
			ratio = width/height
			if ratio < 0.7 or ratio > 1.3: #not a square
				continue
			
			filtered_boxes.append(b)
		
		signs = []
		
		#now calculate HOG-Descriptor for every Box
		for b in filtered_boxes:
			cropped = img[b[0]:b[2]+1,b[1]:b[3]+1]
			scaled = transform.resize(cropped, (SIZE_X, SIZE_Y), anti_aliasing=True, mode="constant")
			desc = feature.hog(scaled, pixels_per_cell=(6,6), cells_per_block=(2,2), visualize=False, block_norm="L1", transform_sqrt=True)
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
				score = self.overlap(s, k)
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

