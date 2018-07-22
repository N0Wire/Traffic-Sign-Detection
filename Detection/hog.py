import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform, feature, exposure
import pandas as pd

###########################################
"""
author: Kim-Louis Simmoteit

This script is used to visualize preprocessor
settings.
"""
###########################################

path = "../Data/GTSRB/Final_Training/Images"
classes = [0, 14, 22, 27]

index = 1
desc_len = 0

images = []
hogs = []

for i in classes:
	full_path = "{}/{:05d}/".format(path, i)
	csvfile = "{}GT-{:05d}.csv".format(full_path, i)
	
	#load data
	data = pd.read_csv(csvfile, delimiter=";")
	names = data["Filename"]
	widths = data["Width"]
	heights = data["Height"]
	
	img_path = full_path + names[9]
	img = io.imread(img_path)
	img = transform.resize(img, (64,64), anti_aliasing=True)
	images.append(img)
	
	desc, h = feature.hog(img, pixels_per_cell=(6,6), cells_per_block=(2,2), visualize=True, block_norm="L1", transform_sqrt=True, multichannel=True) # cell(6,6); block(2,2
	h = exposure.rescale_intensity(h, in_range=(0, 10))
	hogs.append(h)

index = 1
plt.figure(1, dpi=100, figsize=(6,3))
for j in range(4):
	plt.subplot(2, 4, index)
	plt.axis("off")
	plt.imshow(images[j])
	index += 1
	
for j in range(4):
	plt.subplot(2, 4, index)
	plt.axis("off")
	plt.imshow(hogs[j], cmap=plt.cm.gray)
	index += 1

plt.tight_layout()
plt.savefig("Plots/HOG.png")

