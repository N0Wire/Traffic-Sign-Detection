# -*- coding: utf-8 -*-
"""
Object Recognition and Image Understanding
Prof. Bjoern Ommer
SS18

Project

@author: Oliver Drozdowski
"""

from scipy import misc
import os
from matplotlib import pyplot as plt
    
# IMPORT PROJECT FILES
from Classifier.dataloader import dataset

if __name__ == "__main__":
    
    # Set up all the necessary paths for importing data
    filepath_this_file = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(filepath_this_file + "/Data")
    filepath_train = os.path.join(data_path + "/GTSRB/Final_Training/Images")
    filepath_test = os.path.join(data_path + "/GTSRB/Final_Test/Images")
    
    #trainset = dataset(filepath_train, split="train")
    #trainset.subset(100, False)
    
    #print(len(trainset))
    #for i in range(10):
    #    print(trainset[i])
    
    # Load the testset
    testset = dataset(filepath_test, split="test")
    
    # Use subset of testset
    testset.subset(0.1, True)
    
    
    """
    Here we create the images of the preprocessor used in the report.
    """
    
    fig = plt.figure(figsize=(9.6,5), dpi=100)
    for j in range(5):
        
        original = misc.imread(testset[j]['filepath'])
        img = testset[j]['numpy'][...,:3]
        canny = testset[j]['numpy'][...,3]
        
        
        fig.add_subplot(3,5,j+1)
        plt.imshow(original)
        plt.xticks([]), plt.yticks([])
        plt.title("Original Image")    
        
        fig.add_subplot(3,5,j+6)
        plt.imshow(img)
        plt.title("Preprocessed Image")    
        plt.xticks([]), plt.yticks([])
        
        fig.add_subplot(3,5,j+11)
        plt.imshow(canny, cmap="gray")
        plt.title("Canny Image")    
        plt.xticks([]), plt.yticks([])
        
    fig.tight_layout()

    plt.show(fig)
    fig.savefig('./Classifier/Plots/preprocessor.pdf', dpi=300)
   