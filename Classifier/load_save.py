# -*- coding: utf-8 -*-
"""
Object Recognition and Image Understanding
Prof. Bjoern Ommer
SS18

Project

@author: Oliver Drozdowski
"""

import torch
import numpy as np
import torch.nn as nn
import torchvision

# IMPORT PROJECT FILES
from dataloader import dataset

class Logger:
    """
    The class Logger logs the losses, batch accuracies, trainset accuracies and
    testset accuracies during traing as a function of epoch/trainstep.
    The data can be saved and loaded to a file to simplify evaluation afterwards and create
    plots after traintime.
    """
    def __init__(self):
        # Empty constructor
        self.loss_list = []
        self.batch_acc_list = []
        self.train_acc_list = []
        self.test_acc_list = []
    
        return None
    
    def save(self, filename):
        """
        Save all data into one .npz file
        Arguments:  filename - filename without .npz
        """
        loss = np.array(self.loss_list)
        batch = np.array(self.batch_acc_list)
        train = np.array(self.train_acc_list)
        test = np.array(self.test_acc_list)
        np.savez(filename + ".npz", loss=loss, batch=batch, train=train, test=test)
        
        return None
    
    def load(self, filename):
        """
        Read all data from one .npz file created with Logger.save
        Arguments:  filename - filename without .npz
        """
        loaded_data = np.load(filename + ".npz")
        self.loss_list = loaded_data['loss']
        self.batch_acc_list = loaded_data['batch']
        self.train_acc_list = loaded_data['train']
        self.test_acc_list = loaded_data['test']
        
        return None
    
def save_stn_data(model, filename):
    """
    Saves the list of output images of stn network model.list_thetas to file.
    Arguments:  model - CNN_STN object
                filename - filename without .npy
    """
    # Get original image and append to list
    data = model.databatch.clone()
    input_tensor = data.data
    out_grid = convert_image_np(torchvision.utils.make_grid(input_tensor.narrow(1,0,3)).cpu())
    
    
    list_imgs = [out_grid] + model.list_thetas
    # Convert list of images into single numpy array
    save_array = np.stack(list_imgs).astype(np.uint8)
    np.save(filename + ".npy", save_array)
    
    return None

def load_stn_data(filename, model=None):
    """
    Loads the list of output images of stn network model.list_thetas to file.
    The output is given as a list of numpy_arrays which correspond to grid images
    Arguments:  filename - filename without .npy
                model - if specified the data will be written directly into model.list_thetas
    """
    loaded_array = np.load(filename + ".npy")
    total_list = loaded_array.tolist()
    list_img = []
    for img in total_list:
        list_img.append(np.array(img).astype(np.uint8))
    if model is not None:
        model.list_thetas = list_img
    
    return list_img


def save_stn_cnn(model, filename):
    """
    Save all weights and attributes of model (CNN_STN instance) to several files
    Arguments:  model - CNN_STN instance to be saved
                filename - The basefilename. The files will be saved as filename_**.**
    """
    # Save weights
    torch.save(model.state_dict(), filename + '.ckpt')
    
    # Save example grid of images
    torch.save(model.databatch, filename + "_db.pt")
    
    # Save STN examples (list_thetas)
    save_stn_data(model, filename + "_stnd.pt")
    
    # Save list_train_acc and list_test_acc
    train = model.list_train_acc
    save_array = np.stack(train)
    np.save(filename + "_tr.npy", save_array)
    
    test = model.list_test_acc
    save_array = np.stack(test)
    np.save(filename + "_ts.npy", save_array)
    
    return None

def load_stn_cnn(model, filename):
    """
    Load all weights and attributes of model (CNN_STN instance) from several files
    created with the save_stn_cnn function.
    Arguments:  model - CNN_STN instance where to load the attributes into
                filename - The basefilename. The files are saved as filename_**.**
    """
    # Load weights
    model.load_state_dict(torch.load(filename + '.ckpt'))
    
    # Load example grid of images
    model.databatch = torch.load(filename + "_db.pt")
    
    # Load STN examples (list_thetas)
    load_stn_data(filename + "_stnd.pt", model)
    
    # Load list_train_acc and list_test_acc
    train_array = np.load(filename + "_tr.npy")
    train_list = train_array.tolist()
    model.list_train_acc = train_list

    test_array = np.load(filename + "_ts.npy")
    test_list = test_array.tolist()
    model.list_test_acc = test_list
    
    return None


def convert_image_np(tensor):
    """
    Convert a Tensor with arbitrary channel number (e.g. 4 with canny edge) to numpy image.
    Arguments:  tensor - tensor of image or batch of images
    """
    np_array = tensor.numpy()
    np_array = np.moveaxis(np_array,0,-1)
    np_array = np_array[...,:3]

    return np_array.astype(np.uint8)
    