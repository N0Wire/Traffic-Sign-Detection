import numpy as np
from sign_checker import SignChecker

#load all calculated HOG-Descriptors
path = "Data/"
max_run = 60
data = np.load(path+"gtsdb.npy")
for i in range(1,max_run+1):
	if i==58: #58 has no sign and no negative samples -> neglect
		continue
	data = np.concatenate((data, np.load(path+"hog_run_"+str(i)+".npy")), axis=0)

for i in range(43):
	data = np.concatenate((data, np.load(path+"hog_train_c"+str(i)+".npy")), axis=0)

print("Data loaded! Training SVM ...")
sc = SignChecker(10.0)
sc.train(data[:,1:],data[:,0])
sc.save("SVMs/7_")

#parameters:
#1 - C=10 gamma=5 #start 16:40 (forgot transform_sqrt at full image!)	->RIP
#2 - C=1.0 gamma="auto" - standard settings									
#3 - C=1.0 gamma=0.1
#4 - same settings as 3 just more data									->RIP
#5 - same settings as 2 just more data									->RIP
#6 - C=2.0 gamma="auto"													-> slightly better
#7 - C=10 gamma="auto"
