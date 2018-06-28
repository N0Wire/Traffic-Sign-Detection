import numpy as np
from skimage import segmentation
from regions import Region


#TODO: -> Calc and handle Similarities/Merging
#TODO: -> Think of way to save/return bounding boxes
#TODO: -> Create Class SignChecker -> contains SVM and checks if bounding box contains a traffic sign
#TODO: -> Before evaluating bounding box with SVM check aspect ratio to pre select uncertain cases (eg, very long or very high boxes can't contain a sign!)
#TODO: -> Look for a way to only return 1 bounding box per sign

class SelectiveSearch:
	def __init__(self):
		self.segment = 0					#segmented picture
		self.im_size = 0
		self.num_classes = 0 				#total number of classes
		self.regions = [] 					#array of Region elements
		self.similarities = np.zeros((1,3)) #Similarities 0: i, 1: j, 2: S
	
	#calculate Similarity between Region1 and Region2
	#r1, r2: Region object
	#a: which similarities are used (only 0 or 1.0)
	def similarity(self, r1, r2, a=[1, 1, 1, 1]):
		s_color = np.sum(np.minimum(r1.colordesc, r2.colordesc))
		s_texture = np.sum(np.minimum(r1.texturedesc, r2.texturedesc))
		s_size = 1.0-(r1.num_pix+r2.num_pix)/self.im_size
		
		#tight bounding box
		tl = np.minimum(r1.bbox[0],r2.bbox[0])
		tr = np.minimum(r1.bbox[1],r2.bbox[1])
		bl = np.maximum(r1.bbox[2],r2.bbox[2])
		br = np.maximum(r1.bbox[3],r2.bbox[3])
		
		s_fill = 1.0-((tl-bl)(*tr-br)-r1.num_pixels-r2.num_pixels)/self.im_size
		return (float(a[0])*s_color + float(a[1])*s_texture + float(a[2])*s_size + float(a[3])*s_fill)
	
	#Merge two Regions R1 and R2 -> propagate histograms, new bounding box, new neighbours, ... -> calc new similarity
	def merge(self, r1, r2):
		return 0
	
	#initial step
	def init_step(self, img, k=100, minsize=50):
		self.im_size=img.shape[0]*img.shape[1]

		#create segmentation
		self.segment = segmentation.felzenszwalb(img, k, 0.8, minsize)
		self.num_classes = np.amax(self.segment)+1 # number of classes
		
		#create regions arrays
		self.regions = []
		for i in range(self.num_classes):
			r = Region(i)
			r.evaluate(img, self.segment)
			self.regions.append(r)
		
		#calculate similiarities s(ri,rj)
	
	#go level up in the hierachy (merge highest similarity)
	def step(self):
		test = 0
