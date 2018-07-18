import numpy as np
import matplotlib.pyplot as plt
import os



###########################################
"""
author: Kim-Louis Simmoteit

This script is used to evaluate the localization
with using data from collect_objects.py.
"""
###########################################

#loads all data needed for localization (all 900 images)
path_load = "./Runs/"
path_output = "./Plots/"

if not os.path.exists(path_output):
	os.makedirs(path_output)

best_overlapps = []
times = []

num_zeros = 0

max_runnum = 5 #90
for run_num in range(1, max_runnum+1):
	bos = np.load(path_load+"overlap_"+str(run_num)+".npy")
	ts = np.load(path_load+"time_"+str(run_num)+".npy")

	best_overlapps.append(bos)
	for b in bos:
		if b==0:	#no overlapp
			num_zeros += 1
	times.append(ts)

best_overlapps = np.concatenate(best_overlapps)
times = np.concatenate(times)

mbo = np.sum(best_overlapps)/float(best_overlapps.shape[0])	#mean best overlap
mt = np.sum(times)/float(times.shape[0])					#mean time
recall = float(best_overlapps.shape[0]-num_zeros)/(best_overlapps.shape[0])

print("Mean-Best-Overlap: " + str(mbo))
print("Mean Time: " + str(mt))
print("Recall: " + str(recall))

#show histograms
plt.figure(1)
plt.title("Best Overlap(IoU)-Histogram")
plt.xlabel("IoU")
plt.ylabel("#Entries")
plt.hist(best_overlapps, bins=25, label="IoU", edgecolor="black")
plt.axvline(x=mbo, color="r", label="Mean={:.3f}".format(mbo))
plt.legend()
plt.savefig(path_output+"iou_histogram.pdf")

plt.figure(2)
plt.title("Time-Histogram")
plt.xlabel("Time [s]")
plt.ylabel("#Entries")
plt.hist(times, bins=25, label="Time", edgecolor="black")
plt.axvline(x=mt, color="r", label="Mean={:.3f}".format(mt))
plt.legend()
plt.savefig(path_output+"time_histogram.pdf")
