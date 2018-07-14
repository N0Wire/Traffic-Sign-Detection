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
from network_utils import convert_image_np

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
    save_array = np.stack(list_imgs)
    np.save(filename + ".npy", save_array)
    
    return None

def load_stn_data(filename):
    """
    Loads the list of output images of stn network model.list_thetas to file.
    The output is given as a list of numpy_arrays which correspond to grid images
    Arguments:  filename - filename without .npy
    """
    loaded_array = np.load(filename + ".npy")
    total_list = loaded_array.tolist()
    list_img = []
    for img in total_list:
        list_img.append(np.array(img))
        
    return list_img


def save_stn_cnn(model, filename):
    torch.save(model.state_dict(), filename + '.ckpt')
    return None

def load_stn_cnn(model, filename):
    model.load_state_dict(torch.load(filename + '.ckpt'))
    return None