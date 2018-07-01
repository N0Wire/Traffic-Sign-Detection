# -*- coding: utf-8 -*-
"""
Object Recognition and Image Understanding
Prof. Bjoern Ommer
SS18

Project

@author: Kim-Louis Simmoteit, Oliver Drozdowski
"""

from scipy import misc
import numpy as np
import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from random import shuffle
from skimage import exposure

class preprocessor:
    """
    The preprocessor is created using all training images. It calculates
    different values and has a method to preprocess an image file.
    """
    def __init__(self, list_of_filepaths, do_canny=True, split="train"):
        self.list_of_filepaths = list_of_filepaths
        self.split = split
        """
        self.means = 0
        self.percentiles = None
        
        if split == "train":
            # Calculate mean of all images for all images and the mean value for
            # the 2- and 98-percientile of all images to bring all images to scale
            image_means = []
            image_percentiles = []
            for path in list_of_filepaths:
                image = misc.imread(path)
                channel_means = []
                channel_percentiles = []
                for channel in image:
                    # Calculate mean for every channel seperately
                    mean = np.mean(channel)
                    channel_means.append(mean)
                    # Calculate percentiles for every channel
                    percentile_2, percentile_98 = np.percentile(channel, (2, 98))
                    channel_percentiles.append([percentile_2,percentile_98])
            
                image_means.append(channel_means)
                image_percentiles.append(channel_percentiles)
            
            mean_red, mean_green, mean_blue = np.mean(image_means, axis=0)
            means = [mean_red, mean_green, mean_blue]
            percentiles = np.mean(image_percentiles, axis=0)
            print(mean_red)
        """          

    def calc_tensor(self, image):
        """
        # First we want to get the intensities to the mean values from the trainset
        for i, channel in enumerate(image):
            # Rescale means to mean determined in testimages
            diff_mean = np.mean(channel)-means[i]
            channel = channel - diff_mean
            # Rescale range to mean range in testimages
            p2, p98 = np.percentile(channel, (2, 98))
            diff_p2 = p2 - percentiles[i][0]
            diff_p98 = p98 - percentiles[i][1]
            channel = exposure.rescale_intensity(channel, in_range=(p2, p98), out_range=(p2-diff_p2, p98-diff_p98))                
            image[i] = channel
        """
                     
        # On these normalized sets we equalize the histoograms to improve contrast         
        for i, channel in enumerate(image):         
            channel_eq = exposure.equalize_hist(channel)
            image[i]=channel
                
        # Calculate Canny Edge detector
        
        tensor = torch.Tensor(image).float()
        return tensor

class image:
    """
    
    tensor:     the preprocessed tensor of image
    class_id:   the groundtruth class id of image
    """
    def __init__(self, filepath, classid, preprocessor):
        
        self.filepath = filepath
        self.classid = classid
        image = misc.imread(self.filepath)
        self.tensor = preprocessor.calc_tensor(image)
        
    def __str__(self):
        
        return str(self.filepath) + " ; " + str(self.classid)
        
class dataset(Dataset):
    """
    The dataset is initialized with the filepaths and the split.
    It is based on the Dataset class of pytorch.
    It saves a list of all elements of split in datatype image.
    The dataset can be subsampled randomly to decrease the amount of data
    """
    def __init__(self, filepath, split="train", preproc=None):
        """
        Arguments:  filepath is path to /Images directory of dataset split
                    split is either train or test of dataset
        """
        self.filepath = filepath
        
        self.list_paths = []
        self.list_images = []
        self.list_labels = []

        self.preprocessor = preproc
        self.numelements = None
        self.split = split
        
        if split == "train":
            # Generate trainlist of paths and labels
            for dirpath, dirnames, filenames in os.walk(filepath):
                dirnames.sort()
                filenames.sort()
                files = [ fi for fi in filenames if fi.endswith(".ppm") ]
                for filename in files:
                    path_to_img = os.path.join(dirpath, filename)
                    class_id = int(os.path.basename(os.path.normpath(dirpath)))
                    self.list_paths.append(path_to_img)
                    self.list_labels.append(class_id)
        
            # Generate preprocessor
            if self.preprocessor is None:
                self.preprocessor = preprocessor(self.list_paths, split)
                    
            # Generate trainlist of images
            for i, path in enumerate(self.list_paths):
                self.list_images.append(image(path, self.list_labels[i], self.preprocessor))
    
        else:
            # Generate testlist of paths
            groundtruth_file = []
            for dirpath, dirnames, filenames in os.walk(filepath_test):
                filenames.sort()
                files = [ fi for fi in filenames if fi.endswith(".ppm") ]
                groundtruth_file_temp = [fi for fi in filenames if fi.endswith(".csv")]
                groundtruth_file.append(os.path.join(dirpath,groundtruth_file_temp[0]))
                for filename in files:
                    path_to_img = os.path.join(dirpath, filename)
                    self.list_paths.append(path_to_img)
                    
            # Generate testlist of labels
            gt = pd.read_csv(groundtruth_file[0], delimiter=";")
            self.list_labels = gt['ClassId']
        
            # Generate preprocessor
            self.preprocessor = preprocessor(self.list_paths, split)
                    
            # Generate testlist of images
            for i, path in enumerate(self.list_paths):
                self.list_images.append(image(path, self.list_labels[i], self.preprocessor))
    
        print("Done importing data from split: " + split)
        return None
    
    def __getitem__(self, idx):
        return self.list_images[idx]

    def __len__(self):
        return len(self.list_labels)
    
    def rand_subsample(self, size):
        """
        rand_subsample randomizes the ordering of the dataset and then sets the dataset to
        a subsampled version of the dataset.
        Arguments:  size - size of subsample
        """
        data = list(zip(self.list_images, self.list_paths, self.list_labels))
        shuffle(data)
        im_rand, pa_rand, la_rand = zip(*data)
        self.list_images = list(im_rand)[:size]
        self.list_paths = list(pa_rand)[:size]
        self.list_labels = list(la_rand)[:size]
        
        return None
    
    def subset(self, im_per_class, fractional=False):
        """
        subset draws a subset of the dataset and then sets the dataset itself to
        this subset. The subset is selected for the trainset such that im_per_class number of images
        per class are selected instead of the full set. If there are less than im_per_class
        images for a class, the maximum number is used. For the testset the first im_per_class images
        are chosen.
        
        Arguments:  im_per_class - no. of images per class
                    factional - boolean stating whether im_per_class is fractional in [0,1]
                                i.e. im_per_class=0.8 means 80% of images for True
        """
        # Train subset
        if self.split == "train":
            if fractional:
                i=0
                class_0 = self.list_labels[0]
                while self.list_labels[i] == class_0:
                    i+=1
            
                im_per_class = int(im_per_class * i)
        
            index_list = []
            class_img = self.list_labels[0]
            counter_label = 0
            for i, label_i in enumerate(self.list_labels):    
                # We sample only im_per_class images per class
                if counter_label < im_per_class:
                    # If new image is in correct class, add to list
                    if class_img == label_i:                   
                        index_list.append(i)
                        counter_label += 1
                        # If new image is not in correct class, we start the counting again
                    else:
                        index_list.append(i)
                        counter_label = 1
                        class_img = label_i
                        # If we are above the im_per_class images for class, we change class
                else:
                    if class_img != label_i:
                        index_list.append(i)
                        counter_label = 1
                        class_img = label_i
                    
            self.list_images = [self.list_images[i] for i in index_list]
            self.list_paths = [self.list_paths[i] for i in index_list]
            self.list_labels = [self.list_labels[i] for i in index_list]
        # Test subset    
        else:
            if fractional:
                im_per_class = int(im_per_class*len(self.list_labels))
            if im_per_class >= len(self.list_labels):
                return None
            self.list_images = self.list_images[:im_per_class]
            self.list_paths = self.list_paths[:im_per_class]
            self.list_labels = self.list_labels[:im_per_class]
            
            return None            
            
                
                
# Testing stuff
        
if __name__ == "__main__":
    filepath_this_file = os.path.dirname(os.path.abspath(__file__))
    filepath_train = os.path.join(filepath_this_file + "/GTSRB/Final_Training/Images")
    filepath_test = os.path.join(filepath_this_file + "/GTSRB/Final_Test/Images")
    """
    trainset = dataset(filepath_train, split="train")
    trainset.subset(100, False)
    
    print(len(trainset))
    for i in range(10):
        print(trainset[i])
    """
    testset = dataset(filepath_test, split="test")
    
    print(len(testset))
    for i in range(10):
        print(testset[i])
    
    testset.subset(0.1, True)
    
    print(len(testset))
    for i in range(10):
        print(testset[i])
    