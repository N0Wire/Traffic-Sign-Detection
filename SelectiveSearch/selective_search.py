import numpy as np
import scipy.ndimage as si
from skimage import color,segmentation
from regions import Region
from operator import itemgetter
import time


class SelectiveSearch:
	def __init__(self):
		self.segment = 0					#segmented picture
		self.im_size = 0
		self.num_classes = 0 				#total number of classes
		self.regions = [] 					#array of Region elements
		self.similarities = [] 				#Similarities 0: i, 1: j, 2: S
		self.bboxes = []					#bounding boxes
		self.num_regions = 0				#counter for regions
		
		self.visualize = False				#Visualize Selective Search
	
	#Find Neighbours for region r and also add r as neighbour
	#needs to be called after regions are created
	def find_neighbours(self, r):
		#crop out bounding box (choose one bigger -> look for neighouring class labels)
		tl = r.bbox[0] #top left
		bl = r.bbox[2] # bottom left
		tr = r.bbox[1] #top right
		br = r.bbox[3] #bottom right
		
		if tl > 0:
			tl -= 1
		if tr > 0:
			tr -= 1
		if bl < (self.segment.shape[0]-1):
			bl += 1
		if br < (self.segment.shape[1]-1):
			br += 1
		
		crop = self.segment[tl:bl+1,tr:br+1]
		"""
		for i in range(crop.shape[0]):
			for j in range(crop.shape[1]):
				cur = crop[i][j]
				if cur != r.id:
					#check for neighbours
					is_neighbour = False
					
					if (i-1) >= 0 and not is_neighbour:
						is_neighbour = (crop[i-1][j]==r.id)
					if (j-1) >= 0 and not is_neighbour:
						is_neighbour = (crop[i][j-1]==r.id)
					if (i+1) <= (crop.shape[0]-1) and not is_neighbour:
						is_neighbour = (crop[i+1][j]==r.id)
					if (j+1) <= (crop.shape[1]-1) and not is_neighbour:
						is_neighbour = (crop[i][j+1]==r.id)
					
					#append if not exists
					if is_neighbour:
						if cur not in r.neighbours:
							r.neighbours.append(cur)

		"""
		#https://stackoverflow.com/questions/38073433/determine-adjacent-regions-in-numpy-array
		y = crop == r.id  # convert to Boolean

		rolled = np.roll(y, 1, axis=0)          # shift down
		rolled[0, :] = False             
		z = np.logical_or(y, rolled)

		rolled = np.roll(y, -1, axis=0)         # shift up 
		rolled[-1, :] = False
		z = np.logical_or(z, rolled)

		rolled = np.roll(y, 1, axis=1)          # shift right
		rolled[:, 0] = False
		z = np.logical_or(z, rolled)

		rolled = np.roll(y, -1, axis=1)         # shift left
		rolled[:, -1] = False
		z = np.logical_or(z, rolled)

		r.neighbours = set(np.unique(np.extract(z, crop))) - set([r.id])
	
	
	#calculate Similarity between Region1 and Region2
	#r1, r2: Region object
	#a: which similarities are used (only 0 or 1.0)
	def similarity(self, r1, r2, a=[1, 1, 1, 1]):
		s_color = np.sum(np.minimum(r1.colordesc, r2.colordesc))
		s_texture = np.sum(np.minimum(r1.texturedesc, r2.texturedesc))
		s_size = 1.0 - float(r1.num_pix+r2.num_pix)/self.im_size
		
		#tight bounding box
		tl = np.minimum(r1.bbox[0], r2.bbox[0])
		tr = np.minimum(r1.bbox[1], r2.bbox[1])
		bl = np.maximum(r1.bbox[2], r2.bbox[2])
		br = np.maximum(r1.bbox[3], r2.bbox[3])
		
		s_fill = 1.0 - float((tl-bl)*(tr-br)-r1.num_pix-r2.num_pix)/self.im_size
		return (float(a[0])*s_color + float(a[1])*s_texture + float(a[2])*s_size + float(a[3])*s_fill)
	
	#initial step
	def init_step(self, img, col="hsv", k=50, minsize=50, avec = [1,1,1,1]): #minsize=20
		self.regions = []
		self.similarities = []
		self.bboxes = []
	
		if col=="hsv":
			img = color.rgb2hsv(img) #convert to hsv
		elif col=="lab":
			img = color.rgb2lab(img) #convert to lab
		#default: RGB
		else:
			img = img.astype(float) #to prevent overflow
			
		self.im_size=float(img.shape[0]*img.shape[1])

		#create segmentation
		self.segment = segmentation.felzenszwalb(img, k, 0.8, minsize)
		self.num_classes = np.amax(self.segment)+1 			# number of classes
		#print("Starting with " + str(self.num_classes) + " classes!")
		self.num_regions = self.num_classes
		
		#calc gradient maps
		gradrx = si.filters.gaussian_filter(img[:,:,0], 1.0, order=(0,1), mode="nearest")
		gradgx = si.filters.gaussian_filter(img[:,:,1], 1.0, order=(0,1), mode="nearest") 
		gradbx = si.filters.gaussian_filter(img[:,:,2], 1.0, order=(0,1), mode="nearest")
		gradry = si.filters.gaussian_filter(img[:,:,0], 1.0, order=(1,0), mode="nearest")
		gradgy = si.filters.gaussian_filter(img[:,:,1], 1.0, order=(1,0), mode="nearest")
		gradby = si.filters.gaussian_filter(img[:,:,2], 1.0, order=(1,0), mode="nearest")
		
		gradmap_r = np.sqrt(gradrx**2+gradry**2)
		gradmap_g = np.sqrt(gradgx**2+gradgy**2)
		gradmap_b = np.sqrt(gradbx**2+gradby**2)
		#use np.arctan2 instead of arctan to handle 0 gradients
		anglemap_r = np.arctan2(gradry,gradrx)*180/np.pi
		anglemap_g = np.arctan2(gradgy,gradgx)*180/np.pi
		anglemap_b = np.arctan2(gradby,gradbx)*180/np.pi
		
		#create regions arrays
		self.regions = []
		for i in range(self.num_classes):
			r = Region(i)
			r.evaluate(img, self.segment, gradmap_r, gradmap_g, gradmap_b, anglemap_r, anglemap_g, anglemap_b)
			self.regions.append(r)
			self.bboxes.append(r.bbox)
		
		#calc region neighbours
		for r in self.regions:
			self.find_neighbours(r)
		
		#calculate similiarities s(ri,rj)
		for i,r in enumerate(self.regions):
			for n in r.neighbours:
				temp = [i, n, self.similarity(r, self.regions[n])]
				temp2 = [n, i, temp[2]]
				#check if already calculated
				if temp2 not in self.similarities:
					self.similarities.append(temp)
		self.similarities = sorted(self.similarities, key=itemgetter(2), reverse=True)
	
	#Merge two Regions R1 and R2 -> propagate histograms, new bounding box, new neighbours, ... -> calc new similarity
	def merge(self, r1, r2):
		newr = Region(self.num_regions)
		
		#propagate informations
		newr.num_pix = r1.num_pix + r2.num_pix
		newr.colordesc = (r1.colordesc*r1.num_pix+r2.colordesc*r2.num_pix)/newr.num_pix
		newr.texturedesc = (r1.texturedesc*r1.num_pix+r2.texturedesc*r2.num_pix)/newr.num_pix
		
		#bounding box
		tl = np.minimum(r1.bbox[0],r2.bbox[0])
		tr = np.minimum(r1.bbox[1],r2.bbox[1])
		bl = np.maximum(r1.bbox[2],r2.bbox[2])
		br = np.maximum(r1.bbox[3],r2.bbox[3])
		newr.bbox = np.array([tl, tr, bl, br])
		
		#update neighbours
		"""
		for e in r1.neighbours:
			if e not in newr.neighbours:
					newr.neighbours.append(e)
		for e in r2.neighbours:
			if e not in newr.neighbours:
					newr.neighbours.append(e)
		"""
		newr.neighbours = r1.neighbours.union(r2.neighbours)
		
		
		#now recalc neighbour lists of other regions
		for n in newr.neighbours:
			"""
			if r1.id in self.regions[n].neighbours:
				self.regions[n].neighbours.remove(r1.id)
			if r2.id in self.regions[n].neighbours:
				self.regions[n].neighbours.remove(r2.id)
			self.regions[n].neighbours.append(newr.id)
			"""
			self.regions[n].neighbours.discard(r1.id)
			self.regions[n].neighbours.discard(r2.id)
			self.regions[n].neighbours.add(newr.id)
		
		#remove old regions
		"""
		if r1.id in newr.neighbours:
			newr.neighbours.remove(r1.id)
		if r2.id in newr.neighbours:
			newr.neighbours.remove(r2.id)
		"""
		newr.neighbours.discard(r1.id)
		newr.neighbours.discard(r2.id)
		
		return newr
	
	#go level up in the hierachy (merge highest similarity)
	def step(self):
		if len(self.similarities)<2: #we assume, a traffic scene (-> many objects)
			return False
		
		r1 = self.similarities[0][0]
		r2 = self.similarities[0][1]
		
		#if self.regions[int(r1)].merged or self.regions[int(r2)].merged:
		#	print("Something went wrong!")
		#	self.similarities.pop(0)
		#	return True
		
		newr = self.merge(self.regions[int(r1)], self.regions[int(r2)])
		
		#remove old similarites and calc new ones
		self.similarities = [x for x in self.similarities if x[0] != int(r1) and x[0] != int(r2) and x[1] != int(r1) and x[1] != int(r2)]
		
		for n in newr.neighbours:
			temp = [newr.id, n, self.similarity(newr, self.regions[n])] #fast method gives out of range here
			#insert - do better !
			for i,e in enumerate(self.similarities):
				if self.similarities[i][2] < temp[2]:
					self.similarities.insert(i+1, temp)
					break
		
		#if one should visualize also modify segmentation map
		"""
		if self.visualize:
			for i in range(newr.bbox[0], newr.bbox[2]+1):
				for j in range(newr.bbox[1], newr.bbox[3]+1):
					if self.segment[i][j] == r1 or self.segment[i][j] == r2:
						self.segment[i][j] = newr.id
		"""
		
		#print("Merged " + str(r1) + " with " + str(r2))
		#self.regions[int(r1)].merged = True
		#self.regions[int(r2)].merged = True
		self.num_regions += 1
		self.bboxes.append(newr.bbox)
		self.regions.append(newr)
		
		return True
		
		
	#Start SelectiveSearch
	#img = RGB Image
	#method = "single" or "fast", ...
	def run(self, img, method="single"):
		
		bounding_boxes = []
		
		if method == "single":
			self.init_step(img, "hsv", 100)
			ret = True
			num_steps = 0
			
			while ret:
				ret = self.step()
				num_steps += 1
				
				#if (num_steps%100)==0:
				#	print(num_steps)
			
			#print(str(num_steps) + " steps taken for Selective Search!")
			bounding_boxes += self.bboxes
		elif method == "deep": #custom mode -> not in paper
			k_inits = [50, 100]
			for k in k_inits:
				self.init_step(img, "hsv", k)
				ret = True
				num_steps = 0
			
				while ret:
					ret = self.step()
					num_steps += 1
					
				bounding_boxes += self.bboxes			
				#print(str(num_steps) + " steps taken for Selective Search!")
		elif method == "rough": #custom mode
			self.init_step(img, "rgb", 200)
			ret = True
			num_steps = 0
			
			while ret:
				ret = self.step()
				num_steps += 1
				
			bounding_boxes += self.bboxes

		elif method == "fast":
			colors = ["hsv", "lab"]
			k_inits = [50, 100]
			avecs = [[1,1,1,1],[0,1,1,1]]
			
			for c in colors:
				for k in k_inits:
					for a in avecs:
						self.init_step(img, c, k, 20, a)
						ret = True
						while ret:
							ret = self.step()
							
							#append bounding boxes
							bounding_boxes += self.bboxes
		
		return bounding_boxes

