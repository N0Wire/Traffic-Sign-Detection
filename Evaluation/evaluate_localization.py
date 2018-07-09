import numpy as np
import matplotlib.pyplot as plt

#loads all data needed for localization (all 900 images)
path1 = "Runs/"
path2 = "Evaluation/"
best_overlapps = []
times = []

max_runnum = 70
for run_num in range(1, max_runnum+1):
	bos = np.load(path1+"overlap_"+str(run_num)+".npy")
	ts = np.load(path1+"time_"+str(run_num)+".npy")

	best_overlapps.append(bos)
	times.append(ts)

best_overlapps = np.concatenate(best_overlapps)
times = np.concatenate(times)

mbo = np.sum(best_overlapps)/float(best_overlapps.shape[0])
mt = np.sum(times)/float(times.shape[0])

#show histograms
plt.figure(1)
plt.title("IoU-Histogram")
plt.xlabel("IoU")
plt.ylabel("#Entries")
plt.hist(best_overlapps, bins=25, label="IoU")
plt.axvline(x=mbo, color="r", label="Mean={:.3f}".format(mbo))
plt.legend()
plt.savefig(path2+"iou_histogram.pdf")

plt.figure(2)
plt.title("Time-Histogram")
plt.xlabel("Time [ms]")
plt.ylabel("#Entries")
plt.hist(times, bins=25, label="Time")
plt.axvline(x=mt, color="r", label="Mean={:.3f}".format(mt))
plt.legend()
plt.savefig(path2+"time_histogram.pdf")
