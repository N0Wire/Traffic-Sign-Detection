# -*- coding: utf-8 -*-
"""
Object Recognition and Image Understanding
Prof. Bjoern Ommer
SS18

Project

@author: Oliver Drozdowski
"""
"""
This file can be used to train a CNN+STN classifier and evaluate this model.
It is recommended to run it in Spyder, because we plt.show() figures from the
transformation to control the training process. Images to show scalar values like
Loss, Batchaccuracy, Train and Test Accuracy are plottet to *Project*/Classifier/Plots

In the standard setting the Evaluation is run on our pretrained model such that you can
recreate the plots we have in our project report.

Two Boolean flags in the beginning of main control, whether you want to train a model
(and test this code) or want to evaluate a pretrained model. To evaluate the model you have
trained AFTER running this script with trained=False, you have to change the pretrained
flag in import_classifier.

"""


import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import os
import numpy as np

# IMPORT PROJECT FILES
from Classifier.dataloader import dataset
from Classifier.network_utils import train, evaluate, visualize_stn, visualize_scalar
from Classifier.load_save import Logger, save_stn_data, save_stn_cnn
from Classifier.network import CNN_STN
from Classifier.trainer import import_classifier

if __name__ == "__main__":
    filepath_this_file = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(filepath_this_file + "/Data")
    filepath_train = os.path.join(data_path + "/GTSRB/Final_Training/Images")
    filepath_test = os.path.join(data_path + "/GTSRB/Final_Test/Images")
    
    # If you set trained to False you can train the model. If you want to check the functionality
    # and don't want to train the full model, set functionality to True this simplifies the training
    # on a small subset. You will not override our model, because the filenames don't match.
    trained = True
    functionality = True
    
    
    if not trained:
        """
        If trained is set to False we train our final model with this file. After
        training the model is saved as **_final in folder **Project**/Classifier/Saved
        """
        # Load the datasets for training and evaluation        
        trainset = dataset(filepath_train, split="train")        
        testset = dataset(filepath_test, split="test")
        
        if functionality:
            trainset.subset(0.05, fractional=True)
            testset.subset(0.05, fractional=True)
        
        # Size of datasets
        print("Trainset: " + str(len(trainset)))
        print("Testset: " + str(len(testset)))
        
        # Construct the dataloaders
        dataloader_train = DataLoader(trainset, batch_size=32, shuffle=True)
        dataloader_test = DataLoader(testset, batch_size=32, shuffle=True)
 
        # We want to use CUDA in the training, so check for it       
        use_gpu = torch.cuda.is_available()
        tqdm.write("CUDA is available: " + str(use_gpu))
    
        if not use_gpu:
            print("WARNING! No Cuda available. This will slow down training and is possibly untestet.")
    
        # Create model and logger for all scalar values
        model = CNN_STN(43, use_gpu=use_gpu)
        logger = Logger()
        
        # Set a random batch for grid images and train the model
        model.databatch=next(iter(dataloader_train))["tensor"].cuda()
        train(model, dataloader_train, n_epochs=10, checkpoint_name="training", use_gpu=use_gpu, stn=True, dataloader_test=dataloader_test, logger=logger)
    
        # Evaluate the final accuracies
        print("Train accuracy: " + str(evaluate(model, dataloader_train)))
        print("Test accuracy: " + str(evaluate(model, dataloader_test)))

        # Save the entire model in /Saved
        save_stn_data(model, "./Classifier/Saved/stn_data_final")
        save_stn_cnn(model, "./Classifier/Saved/stn_cnn_final")
        logger.save("./Classifier/Saved/logger_final")

        print("Thank you for training with Deutsche Bahn.")

    else:
        """
        If the model is trained, we use the pretrained model to create all the relevant
        plots for the poster/report.
        """
        # Load the train and testset to calculate the accuracies
        trainset = dataset(filepath_train, split="train")
        testset = dataset(filepath_test, split="test")
        
        # Print size of the test/trainset
        print("Size of Trainset: " + str(len(trainset)))
        print("Size of Testset: " + str(len(testset)))
        
        # Create Dataloaders for evaluate()
        dataloader_train = DataLoader(trainset, batch_size=32, shuffle=True)
        dataloader_test = DataLoader(testset, batch_size=32, shuffle=True)
        
        # Print whether CUDA is available
        use_gpu = torch.cuda.is_available()
        tqdm.write("CUDA is available: " + str(use_gpu))
        
        # Import the pretrained classifier
        model, logger,_ = import_classifier(pretrained=True)
        
        # Print the accuracies on the test and trainset
        print("Train accuracy: " + str(evaluate(model, dataloader_train)))
        print("Test accuracy: " + str(evaluate(model, dataloader_test)))
        
        # Plot the STN results over epochs
        fig = plt.figure(figsize=(9.6,4), dpi=100)
        theta_indices = np.array(model.index_thetas)
        
        epoch_list=[0,6,7,9,11,17,58,59]
        for i in range(8):
            fig.add_subplot(2,4,1+i)
            plt.imshow(model.list_thetas[epoch_list[i]])
            plt.xticks([]), plt.yticks([])
            if i==0: 
                plt.title("Original images")
            elif i >= 1 and i < 6:
                plt.title('Transformed Images\nepoch: ' + str(theta_indices[epoch_list[i],0]+1) + '\ndatabatch: ' + str(theta_indices[epoch_list[i],1]+1))
            else:
                plt.title('Transformed Images\nepoch: ' + str(theta_indices[epoch_list[i],0]+1))
        fig.tight_layout()

        plt.show(fig)
        fig.savefig('./Classifier/Plots/stn_over_epochs.pdf', dpi=300)

        # Visualize the output of the STN of sample grid
        if use_gpu:
            model.cuda()    
        visualize_stn(model)
    
        # Visualize the Loss as function of batches
        visualize_scalar(logger.loss_list, filename="./Classifier/Plots/loss_plot.pdf", title="Loss of total network", xname="batch", 
                         yname="loss", show=False, scalars=1, labels=None, ylim=(0,6))
        
        # Visualize the batch accuracies for all batches
        visualize_scalar(logger.batch_acc_list, filename="./Classifier/Plots/batch_acc_plot.pdf", title="Accuracy of batch", xname="batch", 
                         yname="accuracy", show=False, scalars=1, labels=None)
        
        # Visualize the Train and Test accuracies
        train = np.array(model.list_train_acc)
        test = np.array(model.list_test_acc)
        xlen = train.shape[0]
        ylen = 4
        data = np.zeros((xlen,ylen))
        data[:,0:2] = train
        data[:,2:4] = test
        labels = ["Trainset", "Testset"]
        visualize_scalar(data, filename="./Classifier/Plots/train_test_acc_plot.pdf", title="Accuracy of Datasets", xname="epoch", 
                             yname="accuracy", show=False, scalars=2, labels=labels)
    