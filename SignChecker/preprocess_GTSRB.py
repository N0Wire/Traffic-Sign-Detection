import numpy as np
from skimage import io, transform, feature
import pandas as pd
import multiprocessing as mp

#Take images from GTSRB database and calculate HOG-Descriptors
#-> those are later used to train SVM

#Setting Variables
num_classes = 43 #(0-42)
path_training = "../Data/GTSRB/Final_Training/Images"
path_test = "../Data/GTSRB/Final_Test/Images/"
SIZE_X = 64		#size of final image in x direction
SIZE_Y = 64		#size of final image in y direction
desc_size = 2916 #size of HOG-Descriptor

#use different parts of image, scales them and calculates HOG-Descriptors
#path: path to data-set folder (with / at end)
#name: name of file
#sign_class: class number of sign
def evaluate_image(path, name, bbox):
	descs = []
	full_path = path + name
	
	img = io.imread(full_path)
	
	#full image
	full = transform.resize(img, (SIZE_X, SIZE_Y), anti_aliasing=True, mode="constant")
	desc = feature.hog(full, pixels_per_cell=(6,6), cells_per_block=(2,2), visualize=False, block_norm="L1")
	descs.append(desc)
	
	#bounding box
	crop1 = transform.resize(img[bbox[0]:bbox[2]+1,bbox[1]:bbox[3]+1], (SIZE_X, SIZE_Y), anti_aliasing=True, mode="constant")
	desc = feature.hog(crop1, pixels_per_cell=(6,6), cells_per_block=(2,2), visualize=False, block_norm="L1")
	descs.append(desc)
	
	#cut parts of the sign (always substract about 5 pixels)
	#top
	crop2 = transform.resize(img[bbox[0]+5:,:], (SIZE_X, SIZE_Y), anti_aliasing=True, mode="constant")
	desc = feature.hog(crop2, pixels_per_cell=(6,6), cells_per_block=(2,2), visualize=False, block_norm="L1", transform_sqrt=True)
	descs.append(desc)
	
	#bottom
	crop3 = transform.resize(img[:bbox[2]+1-5,:], (SIZE_X, SIZE_Y), anti_aliasing=True, mode="constant")
	desc = feature.hog(crop3, pixels_per_cell=(6,6), cells_per_block=(2,2), visualize=False, block_norm="L1", transform_sqrt=True)
	descs.append(desc)
	
	#left
	crop4 = transform.resize(img[:,bbox[1]+5:], (SIZE_X, SIZE_Y), anti_aliasing=True, mode="constant")
	desc = feature.hog(crop4, pixels_per_cell=(6,6), cells_per_block=(2,2), visualize=False, block_norm="L1", transform_sqrt=True)
	descs.append(desc)
	
	#right
	crop5 = transform.resize(img[:,:bbox[3]+1-5], (SIZE_X, SIZE_Y), anti_aliasing=True, mode="constant")
	desc = feature.hog(crop5, pixels_per_cell=(6,6), cells_per_block=(2,2), visualize=False, block_norm="L1", transform_sqrt=True)
	descs.append(desc)
	
	#center (only crop 4 pixels)
	crop6 = transform.resize(img[bbox[0]+2:bbox[2]+1-2,bbox[1]+2:bbox[3]+1-2], (SIZE_X, SIZE_Y), anti_aliasing=True, mode="constant")
	desc = feature.hog(crop6, pixels_per_cell=(6,6), cells_per_block=(2,2), visualize=False, block_norm="L1", transform_sqrt=True)
	descs.append(desc)
	
	return descs

####################
#Training Images
print("[+]Training Images")


#preprocess
def process_class(c_index):
	print("Processing Image-Class " + str(c_index))
	full_path = "{}/{:05d}/".format(path_training, c_index)
	csvfile = "{}GT-{:05d}.csv".format(full_path, c_index)
	
	#load data
	data = pd.read_csv(csvfile, delimiter=";")
	names = data["Filename"]
	
	x1 = data["Roi.X1"]
	x2 = data["Roi.X2"]
	y1 = data["Roi.Y1"]
	y2 = data["Roi.Y2"]
	
	#load pictures and preprocess
	descriptors = []
	j = 0
	while j < len(names):
		
		box = [y1[j], x1[j], y2[j], x2[j]]
		descs = evaluate_image(full_path, names[j], box)
		for d in descs:
			temp = [1] #1 for Traffic Sign - 0 for no traffic sign
			for k in d:
				temp.append(k)
			descriptors.append(temp)
		#each track contains 30 images -> take every fifth -> 6 images per track
		j += 5

	cols = ["Class-Label"]
	for c in range(desc_size):
		cols.append("D"+str(c))
	
	#save data
	ds = np.array(descriptors)
	np.save("Data/hog_train_c"+str(c_index)+".npy", ds)


##########################################################
start_values = range(num_classes)

procs = []
for v in start_values:
	p = mp.Process(target=process_class, args=(v,))
	procs.append(p)

for i,p in enumerate(procs):
	print("Start " + str(start_values[i]))
	p.start()

for p in procs:
	p.join()


"""
##################
#Test Images
print("[+]Test Images")

#load data
info = []
data = pd.read_csv(path_test + "GT-final_test.test.csv", delimiter=";")
names = data["Filename"]
x1 = data["Roi.X1"]
x2 = data["Roi.X2"]
y1 = data["Roi.Y1"]
y2 = data["Roi.Y2"]

for j in names:
	box = [y1[j], x1[j], y2[j], x2[j]]
	desc = evaluate_image(path_test, j, box)
	temp = [i, full_path + names[j]]
	for k in desc:
		temp.append(k)
	info.append(temp)

cols = []
for i in range(2916):
	cols.append("D"+str(i))

df = pd.DataFrame(info, columns=cols)
df.to_csv("Data/hog_desc_test.csv")
"""
