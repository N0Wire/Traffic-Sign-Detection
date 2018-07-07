import numpy as np
import scipy.ndimage as si
import matplotlib.pyplot as plt

#TODO: -> Check Texture-Histograms -> maybe use LBP instead
#TODO: -> Normalization of Histograms with L1 [Done]
#TODO: -> Neighbours are not handled correctly
#TODO: -> Check Bounding Boxes

#contains Information for a Region of the segmented image (later on -> own file)
class Region:

	#constructor
	def __init__(self, num):
		self.id = num 						#class-label
		self.bbox = np.array([0, 0, 0, 0]) 	#bounding box
		self.num_pix = 0 					# number of pixels in this region
		self.colordesc = np.zeros(75) 		#color descriptor, contains 75 elements (25 bins per hist)
		self.texturedesc = np.zeros(240) 	#texture descriptor contains 240 elements (10 per color per 8 directions)
		self.neighbours = [] 				#neighbour regions
		self.similarities = []				#List of Similarities of Region pairs
		self.merged = False					#Is Region already merged?
	
	#color histograms (25 bins per color) 
	#pixels = 1d array of pixels belonging to class
	#norm = normalization of histogram
	def color_descriptor(self, pixels):
		r = np.histogram(pixels[:,0], bins=25, range=(0,255))
		g = np.histogram(pixels[:,1], bins=25, range=(0,255))
		b = np.histogram(pixels[:,2], bins=25, range=(0,255))
		
		desc = np.append(r[0], g[0])
		desc = np.append(desc, b[0]).astype(float)
		
		norm = np.sum(np.abs(desc))
		
		return desc/norm
	
	#color histograms (25 bins per color) 
	#pixels = 1d array of pixels belonging to class
	#norm = normalization of histogram
	def texture_descriptor(self, pixels, mask):
		#calc gaussian derivatives
		gradrx = si.filters.gaussian_filter(pixels[:,:,0], 1.0, order=(0,1), mode="nearest")
		gradgx = si.filters.gaussian_filter(pixels[:,:,1], 1.0, order=(0,1), mode="nearest")
		gradbx = si.filters.gaussian_filter(pixels[:,:,2], 1.0, order=(0,1), mode="nearest")
		gradry = si.filters.gaussian_filter(pixels[:,:,0], 1.0, order=(1,0), mode="nearest")
		gradgy = si.filters.gaussian_filter(pixels[:,:,1], 1.0, order=(1,0), mode="nearest")
		gradby = si.filters.gaussian_filter(pixels[:,:,2], 1.0, order=(1,0), mode="nearest")
		
		gradient_r = []
		angle_r = []
		gradient_g = []
		angle_g = []
		gradient_b = []
		angle_b = []
		
		#filter out and looking for neighbour-classes (inefficient :))
		for i in range(pixels.shape[0]):
			for j in range(pixels.shape[1]):
				#check for right class
				if mask[i][j] != self.id:
					#check if it is really a neighbour
					is_neighbour = False
					
					#only check bottom, top, left,right
					if (i-1)>=0:
						is_neighbour = (mask[i-1][j] == self.id)
					if (i+1)<=(pixels.shape[0]-1):
						is_neighbour = (mask[i+1][j] == self.id)
					if (j-1)>=0:
						is_neighbour = (mask[i][j-1] == self.id)
					if (j+1)<=(pixels.shape[1]-1):
						is_neighbour = (mask[i][j+1] == self.id)
					
					if not is_neighbour:
						continue
				
					if mask[i][j] not in self.neighbours:
						self.neighbours.append(mask[i][j])
					continue
				
				g_r = np.sqrt(gradrx[i][j]**2+gradry[i][j]**2)
				a_r = 90.0
				if gradrx[i][j] != 0:
					a_r = np.arctan(gradry[i][j]/gradrx[i][j])*180.0/np.pi
				g_g = np.sqrt(gradgx[i][j]**2+gradgy[i][j]**2)
				a_g = 90.0
				if gradgx[i][j] != 0:
					a_g = np.arctan(gradgy[i][j]/gradgx[i][j])*180.0/np.pi
				g_b = np.sqrt(gradbx[i][j]**2+gradby[i][j]**2)
				a_b = 90.0
				if gradbx[i][j] != 0:
					a_b = np.arctan(gradby[i][j]/gradbx[i][j])*180.0/np.pi
				
				gradient_r.append(g_r)
				angle_r.append(a_r)
				gradient_g.append(g_g)
				angle_g.append(a_g)
				gradient_b.append(g_b)
				angle_b.append(a_b)
		
		#create 2dhistograms
		r = np.histogram2d(angle_r, gradient_r, bins=[8,10], range=[[-90, 90],[0,255]])
		g = np.histogram2d(angle_g, gradient_g, bins=[8,10], range=[[-90, 90],[0,255]])
		b = np.histogram2d(angle_b, gradient_b, bins=[8,10], range=[[-90, 90],[0,255]])
		
		desc = np.append(r[0].flatten(), g[0].flatten())
		desc = np.append(desc, b[0].flatten()).astype(float)
		
		norm = np.sum(np.abs(desc))
		if norm != 0.0:
			desc = desc/norm
		
		return desc
	
	#calculate region informations (bounding box, color hist, texture hist, num pixels)
	def evaluate(self, img, segmentation):
		mask = (segmentation==self.id)
		pixels = img[mask]
		self.num_pix = len(pixels) #number of pixels in this region
		
		#calculate color histograms
		self.colordesc = self.color_descriptor(pixels)
		
		#bounding box
		cols = np.any(mask, axis=1)
		rows = np.any(mask, axis=0)
		rmin, rmax = np.where(rows)[0][[0, -1]] #row
		cmin, cmax = np.where(cols)[0][[0, -1]] #column
		self.bbox = np.array([cmin, rmin, cmax, rmax])
		
		#crop out bounding box (choose one bigger -> look for neighouring class labels)
		tl = cmin #top left
		bl = cmax+1 # bottom left
		tr = rmin #top right
		br = rmax+1 #bottom right
		
		if tl > 0:
			tl -= 1
		if tr > 0:
			tr -= 1
		if bl < (img.shape[0]-1):
			bl += 1
		if br < (img.shape[1]-1):
			br += 1
		
		cropimg= img[tl:(bl+1),tr:(br+1),:]
		cropmask = segmentation[tl:(bl+1),tr:(br+1)] 
		
		#texture histograms
		self.neighbours = []
		self.texturedesc = self.texture_descriptor(cropimg, cropmask)
