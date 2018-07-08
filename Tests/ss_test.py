import sys
sys.path.append("../SelectiveSearch")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from selective_search import SelectiveSearch

def cut_bbox(img, box, num):
	cropped = img[box[0]:box[2]+1,box[1]:box[3]+1]
	plt.figure(1)
	plt.clf()
	plt.imshow(cropped)
	plt.savefig("ss_test/bbox_" + str(num) + ".png")

img = io.imread("bz_berlin.jpg")#"sign_test.ppm") #mask_test.jpg

ss = SelectiveSearch()

boxes = []
ss.init_step(img, "hsv", 100)
ret = True
num_steps = 0

while ret:
	ret = ss.step()
	num_steps += 1
	
	#after each step save data (similarities, neighbours segments
	"""
	np.savetxt("ss_test/sim_"+str(ss.num_regions)+".txt",ss.similarities)
	
	f = open("ss_test/nei_"+str(ss.num_regions)+".txt", "w")
	for i,n in enumerate(ss.regions):
		f.write(str(i)+"\t"+str(n.neighbours)+"\n")
	f.close()
	
	plt.figure(1)
	plt.clf()
	plt.imshow(ss.segment, cmap=plt.cm.jet)
	plt.colorbar()
	plt.savefig("ss_test/seg_"+str(ss.num_regions)+".png")
	"""

print(str(num_steps) + " steps taken for Selective Search!")
boxes += ss.bboxes




#boxes = ss.run(img)
for i,b in enumerate(boxes):
	cut_bbox(img, b, i)

plt.figure(1)
plt.clf()
plt.axis("off")
plt.imshow(ss.segment, cmap=plt.cm.jet)
plt.colorbar()

for b in boxes:
	#be careful: Rectangel takes ((x,y), width, height)!!!!
	rect = patches.Rectangle((b[1], b[0]),np.abs(b[3]-b[1]), np.abs(b[2]-b[0]), linewidth=1, edgecolor="r", facecolor="none")
	plt.gca().add_patch(rect)

plt.savefig("ss_test/final.png")
plt.show()
