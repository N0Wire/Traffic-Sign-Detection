import numpy as np
import matplotlib.pyplot as plt
import os

###########################################
"""
author: Kim-Louis Simmoteit

This script evaluates data from the test-set of the 
GTSDB dataset for a given SVM.
-> Mean Best Overlap, Precision, Recall
"""
###########################################


#loads all data needed for localization (all 900 images)
svm_prefix = "11_"
path_load = "./Runs/"+svm_prefix[0:len(svm_prefix)-1]+"/"
path_output = "./Plots/"

if not os.path.exists(path_output):
	os.makedirs(path_output)

best_overlapps = []
times = []
num_wbos = 0.0		#number of wrongly detected signs
num_bos = 0.0		#number of best overlapps
num_zeros = 0.0		#number of not detected signs

max_runnum = 75
for run_num in range(61, max_runnum+1):
	bos = np.load(path_load+"overlap_"+str(run_num)+".npy")
	ts = np.load(path_load+"times_"+str(run_num)+".npy")
	wbos = np.load(path_load+"wrongboxes_"+str(run_num)+".npy")
	obos = np.load(path_load+"overlapboxes_"+str(run_num)+".npy")
	num_wbos += float(wbos.shape[0])
	num_bos += float(bos.shape[0])
	
	for b in obos:
		if np.array_equal(b,[0,0,0,0]):	#one sign was not found (no overlapp)
			num_zeros += 1

	best_overlapps.append(bos)
	times.append(ts)

#evaluate
best_overlapps = np.concatenate(best_overlapps)
times = np.concatenate(times)

mbo = np.sum(best_overlapps)/float(best_overlapps.shape[0]) #mean best overlap
mt = np.sum(times)/float(times.shape[0])					#mean time needed
recall = float(num_bos-num_zeros)/num_bos
precision = float(num_bos-num_zeros)/(num_wbos+num_bos-num_zeros)

print("Mean-Best-Overlap: " + str(mbo))
print("Mean Time: " + str(mt))
print("Recall: " + str(recall))
print("Precision: " + str(precision))



#show histograms
plt.figure(1)
plt.title(" Best Overlap(IoU)-Histogram")
plt.xlabel("IoU")
plt.ylabel("#Entries")
plt.hist(best_overlapps, bins=25, label="IoU", edgecolor="black")
plt.axvline(x=mbo, color="r", label="Mean={:.3f}".format(mbo))
plt.legend()
plt.savefig(path_output+"mbo_svm_histogram.pdf")

plt.figure(2)
plt.title("Time-Histogram")
plt.xlabel("Time [s]")
plt.ylabel("#Entries")
plt.hist(times, bins=25, label="Time", edgecolor="black")
plt.axvline(x=mt, color="r", label="Mean={:.3f}".format(mt))
plt.legend()
plt.savefig(path_output+"time_histogram.pdf")
