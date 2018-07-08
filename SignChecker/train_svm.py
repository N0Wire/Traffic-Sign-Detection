import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from skimage import io, transform, feature
from sign_checker import SignChecker

#load bbox data
path = "../Data/FullIJCNN2013/"
infos = pd.read_csv(path + "gt.txt", delimiter=";", header=None)

#loads all data from first 600 runs and and creates histograms and trains SVM
best_overlapps = []
times = []
boxes = []

max_runnum = 5
for run_num in range(1, max_runnum+1):
	bos = np.load("Localization/overlap_"+str(run_num)+".npy")
	ts = np.load("Localization/time_"+str(run_num)+".npy")
	f = open("Localization/boxes_"+str(run_num)+".dat", "rb")
	bs = pickle.load(f)
	f.close()

	best_overlapps.append(bos)
	times.append(ts)
	boxes.append(bs)

best_overlapps = np.concatenate(best_overlapps)
times = np.concatenate(times)

mbo = np.sum(best_overlapps)/float(best_overlapps.shape[0])
mt = np.sum(times)/float(times.shape[0])

#show histograms
plt.figure(1)
plt.title("IoU-Histogram")
plt.xlabel("IoU")
plt.ylabel("#Entries")
plt.hist(best_overlapps, bins=20, label="IoU")
plt.axvline(x=mbo, color="r", label="Mean")
plt.legend()
plt.savefig("Localization/iou_histogram.pdf")

plt.figure(2)
plt.title("Time-Histogram")
plt.xlabel("Time [ms]")
plt.ylabel("#Entries")
plt.hist(times, bins=20, label="Time")
plt.axvline(x=mt, color="r", label="Mean")
plt.legend()
plt.savefig("Localization/time_histogram.pdf")


#crop out image parts
#for i,b in enumerate(boxes):
#	



#sc = SignChecker(1.0)
#sc.train()
#sc.save()
