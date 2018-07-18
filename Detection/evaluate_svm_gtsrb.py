import numpy as np
import pandas as pd
from skimage import io
from sign_checker import SignChecker
import tools

sc = SignChecker()
sc.load("SVMs/4_")

path = "../Data/GTSRB/Final_Test/Images/"

#load infos
info_table = pd.read_csv(path+"GT-final_test.test.csv", delimiter=";")
infos = np.array(info_table)

pred = np.zeros(len(infos))
for i in range(len(infos)):
	full_path = path + infos[i][0]
	img = io.imread(full_path)
	desc = tools.HogDescriptor(img)
	pred[i] = sc.check_desc([desc])[0]

classification = np.sum(pred)/len(infos)
print(classification)
