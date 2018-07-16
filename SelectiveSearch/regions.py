import numpy as np
import scipy.ndimage as si
import matplotlib.pyplot as plt


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
	def texture_descriptor(self, g_r, g_g, g_b, a_r, a_g, a_b):
		#create 2dhistograms
		r = np.histogram2d(a_r, g_r, bins=[8,10], range=[[-90, 90],[0,255]])
		g = np.histogram2d(a_g, g_g, bins=[8,10], range=[[-90, 90],[0,255]])
		b = np.histogram2d(a_b, g_b, bins=[8,10], range=[[-90, 90],[0,255]])
		
		desc = np.append(r[0].flatten(), g[0].flatten())
		desc = np.append(desc, b[0].flatten()).astype(float)
		
		norm = np.sum(np.abs(desc))
		if norm != 0.0:
			desc = desc/norm
		
		return desc
	
	#calculate region informations (bounding box, color hist, texture hist, num pixels)
	def evaluate(self, img, segmentation, g_r, g_g, g_b, a_r, a_g, a_b):
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
		#texture histograms
		gr = g_r[cmin:(cmax+1),rmin:(rmax+1)].flatten()
		gg = g_g[cmin:(cmax+1),rmin:(rmax+1)].flatten() 
		gb = g_b[cmin:(cmax+1),rmin:(rmax+1)].flatten()
		ar = a_r[cmin:(cmax+1),rmin:(rmax+1)].flatten()
		ag = a_g[cmin:(cmax+1),rmin:(rmax+1)].flatten()
		ab = a_b[cmin:(cmax+1),rmin:(rmax+1)].flatten()
		
		self.texturedesc = self.texture_descriptor(gr, gg, gb, ar, ag, ab)

