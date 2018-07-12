import numpy as np
from sign_checker import SignChecker

#load all calculated HOG-Descriptors
path = "Data/"
max_run = 60
data = np.load(path+"hog_run_1.npy")
for i in range(2,max_run+1):
	if i==58: #58 has no sign and no negative samples -> neglect
		continue
	data = np.concatenate((data, np.load(path+"hog_run_"+str(i)+".npy")), axis=0)

for i in range(43):
	data = np.concatenate((data, np.load(path+"hog_train_c"+str(i)+".npy")), axis=0)

print("Data loaded! Training SVM ...")
sc = SignChecker(100.0, 10.0)
sc.train(data[:,1:],data[:,0])
sc.save(path+"1_")

#parameters:
#1 - C=100 gamma=10
