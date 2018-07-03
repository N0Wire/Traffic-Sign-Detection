# -*- coding: utf-8 -*-
"""
Object Recognition and Image Understanding
Prof. Bjoern Ommer
SS18

Project

@author: Kim-Louis Simmoteit, Oliver Drozdowski
This code partially contains code from the solution of exercise 8.
"""

import torch
import os
import torch.nn as nn
from torch.nn import Linear, Conv2d, ReLU, MaxPool2d, CrossEntropyLoss
from torch.nn.init import xavier_normal_, normal_
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms
from dataloader import *

class CNN(nn.Module):
    '''
    Example from exercises
    '''
    def __init__(self, n_classes):
        '''
        Arguments: n_classes -  number of classes
        '''
        super(CNN, self).__init__()
        
        self.conv1 = Conv2d(4, 16, 3, stride=1, padding=1)
        self.conv2 = Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv3 = Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv4 = Conv2d(64, 128, 3, stride=1, padding=1)

        self.maxpool = MaxPool2d(2, 2)

        # input units: (28 / 2 / 2)**2 * 128 = 7**2 * 128 = 2**(2*5) * 2**7 = 2**17
        #               \____________/     \
        #             output tensor size    channels
        self.n_units = 8*8*128
        self.fc1 = Linear(self.n_units, 100)
        self.fc2 = Linear(100, n_classes)

        self.activation = ReLU()

        self.apply(initializer)

    def forward(self, x):
        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        out = self.maxpool(out)

        out = self.conv3(out)
        out = self.activation(out)
        out = self.conv4(out)
        out = self.activation(out)
        out = self.maxpool(out)
        
        out = out.view(-1, self.n_units)
        
        out = self.fc1(out)
        out = self.activation(out)

        self.features = out

        out = self.fc2(out)

        return out


def train(model, dataloader, n_epochs=10, checkpoint_name='training', use_gpu=True):
    '''
    This function trains the CNN model. The training is done for the dataset in the dataloader instance
    for multiple epochs. After every epoch the model is saved.
    
    Arguments:  model - CNN instance
                dataloader - a DataLoader instance based on a dataset or *** instance
                n_epochs - number of epochs to be trained
                checkpoint_name - Name to be specified in the saved model
                use_gpu - Boolean stating whether CUDA shall be used (check first!)
    '''

    if use_gpu:
        model.cuda()
    
    # We use CrossEntropyLoss and the Adam optimizer
    Loss = CrossEntropyLoss()
    Optimizer = torch.optim.Adam(model.parameters())
    
    # We calculate for all epochs
    for epoch in tqdm(range(n_epochs), desc='epoch', position=1):
        
        # We loop through the set of batches
        for batch_index, batch in enumerate(tqdm(dataloader, desc='batch', position=0)):
            train_step = batch_index + len(dataloader)*epoch
            
            # Unpack batch
            images_batch, ids_batch = batch['tensor'], batch['id']
            
            if use_gpu:
                images_batch = images_batch.cuda()
                ids_batch = ids_batch.cuda()

            # Forward
            predictions = model(images_batch)

            # Loss
            loss = Loss(predictions, ids_batch)
            acc = torch.mean(torch.eq(torch.argmax(predictions, dim=-1),
                                      ids_batch).float())
            
            # Zero the gradient before backward propagation
            Optimizer.zero_grad()

            # Backward propagation
            loss.backward()
            
            # Update
            Optimizer.step()

            if train_step % 25 == 0:
                tqdm.write('{}: Batch-Accuracy = {}, Loss = {}'\
                          .format(train_step, float(acc), float(loss)))
        
        # Save the model after every epoch
        torch.save(model.state_dict(), '{}-{}.ckpt'.format(checkpoint_name, epoch))

def initializer(module):
    """
    Initialize the weights with Gaussians (xavier) and bias with normals
    """
    if isinstance(module, Conv2d):
        xavier_normal_(module.weight)
        normal_(module.bias)
    elif isinstance(module, Linear):
        normal_(module.weight)
        normal_(module.bias)


def evaluate(model, dataloader):
    """
    For a CNN model we calculate the accuracy on the dataset of dataloader.
    Arguments:  model - a CNN instance
                dataloader - a dataloader instance bases on dataset or *** class
    """
    acc = 0
    for batch in dataloader:
        images_batch = batch["tensor"]
        predictions = model(images_batch)
        ground_truth = batch["id"]
        acc += torch.mean(torch.eq(torch.argmax(predictions, dim=-1),
                                   ground_truth).float())

    acc /= len(dataloader)

    acc = float(acc.detach().numpy())
    return acc


# Testing stuff
        
if __name__ == "__main__":
    #from matplotlib import pyplot as plt
    filepath_this_file = os.path.dirname(os.path.abspath(__file__))
    filepath_train = os.path.join(filepath_this_file + "/GTSRB/Final_Training/Images")
    filepath_test = os.path.join(filepath_this_file + "/GTSRB/Final_Test/Images")
    
    trainset = dataset(filepath_test, split="test")
    trainset.subset(0.2, fractional=True)
    
    tqdm.write(str(len(trainset)))
    
    dataloader = DataLoader(trainset, batch_size=32, shuffle=True)
    
    labels=[]
    for i in range(len(trainset)):
        labels.append(trainset[i]["id"])
    print(labels)
    
    use_gpu = torch.cuda.is_available()
    tqdm.write("CUDA is available: " + str(use_gpu))
    
    model = CNN(43)
    train(model, dataloader, n_epochs=3, checkpoint_name="test", use_gpu=False)
    print(evaluate(model, dataloader))