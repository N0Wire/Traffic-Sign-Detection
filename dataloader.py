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
from torch.utils.data import Dataset, DataLoader
from skimage.color import rgb2gray
from scipy.ndimage.interpolation import zoom
from random import shuffle
from skimage import exposure
from skimage.feature import canny
from tqdm import tqdm, trange
from time import sleep


class preprocessor:
    """
    The preprocessor takes the images and improves them. It equalizes the histograms
    of all channels for a better contrast and scales the images to the desired size.
    
    Arguments:  size - (y,x) tuple stating the size of processed image
                do_canny -  Boolean stating whether Canny edge is wished (Caveat: Changes in
                            network.py necessary!)
                split - not used as of now
    """
    def __init__(self, size=(32,32), do_canny=True, split="train"):
        self.split = split
        self.ysize_norm, self.xsize_norm = size
        self.do_canny = do_canny
        
    def calc_tensor(self, image):
        
        # We perform histogram stretching to improve contrast         
        
        for i in range(image.shape[-1]):
            channel = image[...,i]
            # Contrast stretching
            p2, p98 = np.percentile(channel, (2, 98))
            channel_eq = exposure.rescale_intensity(channel, in_range=(p2, p98))
            # Override channel
            image[...,i]=channel_eq
        
        # Rescale the image
        channels=[]
        for i in range(image.shape[-1]):
            channel = image[...,i]
            yzoom = self.ysize_norm/channel.shape[0]
            xzoom = self.xsize_norm/channel.shape[1]
            channel = zoom(channel, (yzoom,xzoom))
            channels.append(channel)
        image=np.moveaxis(np.array(channels),(0,1,2),(2,0,1))
        
        # Calculate Canny Edge detector
        if self.do_canny:
            img_gray = rgb2gray(image)
            img_gray = canny(img_gray, sigma=0.5, high_threshold=0.55, low_threshold=0.20, use_quantiles=True)
        
            # Add canny picture as 4th channel
            image = np.concatenate((image,np.expand_dims(img_gray, axis=-1)), axis=-1)

        image=np.moveaxis(image,(0,1,2),(1,2,0))
        #tensor = torch.Tensor(image).float()
        tensor = torch.Tensor(image)
        return tensor

class image:
    """
    Arguements:     filepath - filepath to file
                    classsid - ground truth class id
                    preprocessor - preprocessor object
                    img (optional) - image array, no pathfile has to be given then
                                     (just input something)
                                     
    tensor:     the preprocessed tensor of image
    class_id:   the groundtruth class id of image
    numpy:      gives access to the array-type image
    _data_:     returns data as dict, because DataLoader class needs dicts *sigh*
    """
    def __init__(self, filepath, classid, preprocessor, img=None):
        
        self.filepath = filepath
        self.classid = classid
        if img is None:
            image = misc.imread(self.filepath)
        else:
            image = img
        self.tensor = preprocessor.calc_tensor(image)
        
    def __str__(self):
        # for print function
        return str(self.filepath) + " ; " + str(self.classid)
    
    def numpy(self):
        np_array = self.tensor.numpy()
        np_array = np.moveaxis(np_array,0,-1)
        return np_array.astype(np.uint8)
    
    def _data_(self):
        return {"tensor": self.tensor, "id": self.classid , "filepath": self.filepath, "numpy": self.numpy()}

    
class dataset(Dataset):
    """
    The dataset is initialized with the filepaths and the split.
    It is based on the Dataset class of pytorch.
    It saves a list of all elements of split in datatype image.
    The dataset can be subsampled randomly to decrease the amount of data
    """
    def __init__(self, filepath, split="train", preproc_size=(32,32), preproc_canny=True):
        """
        Arguments:  filepath is path to /Images directory of dataset split
                    split is either train or test of dataset
        """
        self.filepath = filepath
        
        self.list_paths = []
        self.list_images = []
        self.list_labels = []

        self.preproc_size = preproc_size
        self.preproc_canny = preproc_canny
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
            print("Importing and preprocessing data from split: " + split)
            self.preprocessor = preprocessor(size=self.preproc_size, do_canny=self.preproc_canny, split=self.split)
            sleep(1) # sleep for nicer text output
            
            # Generate trainlist of images
            for i, path in enumerate(tqdm(self.list_paths, desc="Imageloader: ")):
                self.list_images.append(image(path, self.list_labels[i], self.preprocessor))
    
        else:
            # Generate testlist of paths
            groundtruth_file = []
            for dirpath, dirnames, filenames in os.walk(filepath):
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
            print("Importing and preprocessing data from split: " + split)
            self.preprocessor = preprocessor(size=self.preproc_size, do_canny=self.preproc_canny, split=self.split)
            sleep(1) # sleep for nicer text output
            
            # Generate testlist of images
            for i, path in enumerate(tqdm(self.list_paths, desc="Imageloader: ")):
                self.list_images.append(image(path, self.list_labels[i], self.preprocessor))
    
        print("Done importing data from split: " + split)
        return None
    
    def __getitem__(self, idx):
        return self.list_images[idx]._data_()

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
    from matplotlib import pyplot as plt
    filepath_this_file = os.path.dirname(os.path.abspath(__file__))
    filepath_train = os.path.join(filepath_this_file + "/GTSRB/Final_Training/Images")
    filepath_test = os.path.join(filepath_this_file + "/GTSRB/Final_Test/Images")
    
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
    
    fig = plt.figure(figsize=(1.5,15), dpi=100)
    for j in range(30):
        
        original = misc.imread(testset[j]['filepath'])
        img = testset[j]['numpy'][...,:3]
        canny = testset[j]['numpy'][...,3]
        
        
        fig.add_subplot(30,3,3*j+1)
        plt.imshow(original)
        plt.xticks([]), plt.yticks([])
        plt.title("Original \n" + str(original.shape), fontsize=3)    
        
        fig.add_subplot(30,3,3*j+2)
        plt.imshow(img)
        plt.title("Preprocessed \n" + str(img.shape), fontsize=3)    
        plt.xticks([]), plt.yticks([])
        
        fig.add_subplot(30,3,3*j+3)
        plt.imshow(canny, cmap="gray")
        plt.title("Canny image", fontsize=3)    
        plt.xticks([]), plt.yticks([])
        
    fig.tight_layout()

    plt.show(fig)
    fig.savefig('hog_test.pdf', dpi=300)


    dataloader = DataLoader(testset, batch_size=32, shuffle=True)
    for i, batch in enumerate(dataloader):
        images_batch, ids_batch = \
                batch['tensor'], batch['id']
        #print(batch)
        print(ids_batch)
        
    """   