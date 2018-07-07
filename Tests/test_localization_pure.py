import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from skimage import io
from selective_search import SelectiveSearch
import time

#dirty method to read .ppm files (need to have format of FulliJCNN2013 images)
def ppm_reader(path):
	f = open(path, "r")
	#"parse" header
	c = f.read(2) #should be P6
	
	f.read(1) # read place holder
	c2 = f.read(4) #1360
	f.read(1)
	c3 = f.read(3) #800
	f.read(1)
	c4 = f.read(3) #255
	f.read(1)
	
	if c != "P6" or c2 != "1360" or c3 != "800":
		print("Can't support Image type!")
		return 0
	
	img = np.zeros((800, 1360, 3), dtype="uint8")
	for i in range(800):
		for j in range(1360):
			for k in range(3):
				img[i][j][k] = np.fromstring(f.read(1), dtype="uint8")
	
	return img

def overlap(box1, box2):
	area1 = float(box1[2]-box1[0])*float(box1[3]-box1[0])
	area2 = float(box2[2]-box2[0])*float(box2[3]-box2[0])
	
	miny = max(box1[0], box2[0])
	minx = max(box1[1], box2[1])
	maxy = min(box1[2], box2[2])
	maxx = min(box1[3], box2[3])
	
	area = float(maxy-miny)*float(maxx-minx)
	
	return area/(area1+area2)

#parse file
path = "../Data/FullIJCNN2013/"
savepath = "Localization/"
infos = pd.read_csv(path + "gt.txt", delimiter=";", header=None)

best_overlapps = np.zeros(len(infos))
times = np.zeros(900)
index = 0
t_index = 0
while index<len(infos):
	j = index
	name = infos[0][index]
	rectangles = []
	overlap_boxes = []
	#look how many bounding boxes are in this picture to find
	while j<len(infos):
		if infos[0][j] == name:
			temp = [infos[2][j], infos[1][j], infos[4][j], infos[3][j]] #identical format, compared to ss_search
			rectangles.append(temp)
			overlap_boxes.append(temp)
		else:
			break
		j += 1
		
	index += len(rectangles)
	
	print("Evaluating " + name)
	#run selective search and obtain bounding boxes
	img = plt.imread(path+name)
	
	start = time.time()
	ss = SelectiveSearch()
	boxes = ss.run(img)# , "fast")
	stop = time.time()
	times[t_index] = stop-start
	t_index += 1
	
	
	for b in boxes:
		#determine best overlapp
		for i,r in enumerate(rectangles):
			score = overlap(r, b)
			if score > best_overlapps[index+i]:
				best_overlapps[index+i] = score
				overlap_boxes[i] = b
	
	for i,r in enumerate(rectangles):
		print("Best-Overlapp: " + str(best_overlapps[index+i]))
	
	plt.figure(1, dpi=100, figsize=(13.6,8.0))
	plt.clf()
	plt.axis("off")
	plt.imshow(img)
	
	for r in rectangles: #given rectangles
		rect = patches.Rectangle((r[1], r[0]),np.abs(r[3]-r[1]), np.abs(r[2]-r[0]), linewidth=1, edgecolor="g", facecolor="none")
		plt.gca().add_patch(rect)
	for b in overlap_boxes: #best matching bounding boxes
		rect = patches.Rectangle((b[1], b[0]),np.abs(b[3]-b[1]), np.abs(b[2]-b[0]), linewidth=1, edgecolor="r", facecolor="none")
		plt.gca().add_patch(rect)
	plt.savefig(savepath + name[0:len(name)-3] + "png") #can't save as ppm
	
	break


#calculate mean best overlap:
mbo = np.sum(best_overlapps)/float(best_overlapps.shape[0])
print("Mean Best Overlapp: " + str(mbo))

mt = np.sum(times)/900.0
print("Mean Time needed for selective search: " + str(mt))
	
