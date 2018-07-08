import numpy as np
from skimage import io, transform, feature
import pandas as pd

#Setting Variables
num_classes = 43 #(0-42)
path_training = "../Data/GTSRB/Final_Training/Images"
path_test = "../Data/GTSRB/Final_Test/Images/"
SIZE_X = 60
SIZE_Y = 60
desc_size = 2916 #size of HOG-Descriptor

#path: path to data-set folder (with / at end)
#name: name of file
#sign_class: class number of sign
def evaluate_image(path, name):
	full_path = path + name
	
	img = io.imread(full_path)
	if img.shape[0] > SIZE_Y or img.shape[1] > SIZE_X:
		img = transform.resize(img, (SIZE_X, SIZE_Y), anti_aliasing=True)
	else:
		img = transform.resize(img, (SIZE_X, SIZE_Y), anti_aliasing=False)
	
	desc = feature.hog(img, pixels_per_cell=(6,6), cells_per_block=(2,2), visualize=False, block_norm="L1")
	return desc

####################
#Training Images
print("[+]Training Images")

info = []

#preprocess
for i in range(num_classes):
	print("Processing Image-Class " + str(i))
	full_path = "{}/{:05d}/".format(path_training, i)
	csvfile = "{}GT-{:05d}.csv".format(full_path, i)
	
	#load data
	data = pd.read_csv(csvfile, delimiter=";")
	names = data["Filename"]
	
	#load pictures and preprocess
	for j in range(len(names)):
		desc = evaluate_image(full_path, names[j])
		temp = [i, full_path + names[j]]
		for k in desc:
			temp.append(k)
		info.append(temp)

cols = ["Class", "Filename"]
for i in range(2916):
	cols.append("D"+str(i))

df = pd.DataFrame(info, columns=cols)
df.to_csv("Data/hog_desc_train.csv")

##################
#Test Images
print("[+]Test Images")

#load data
info = []
data = pd.read_csv(path_test + "GT-final_test.test.csv", delimiter=";")
names = data["Filename"]

for j in names:
	desc = evaluate_image(path_test, j)
	temp = [i, full_path + names[j]]
	for k in desc:
		temp.append(k)
	info.append(temp)

cols = []
for i in range(2916):
	cols.append("D"+str(i))

df = pd.DataFrame(info, columns=cols)
df.to_csv("Data/hog_desc_test.csv")

