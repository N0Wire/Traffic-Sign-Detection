# -*- coding: utf-8 -*-
"""
Object Recognition and Image Understanding
Prof. Bjoern Ommer
SS18

Project

@author: Oliver Drozdowski
"""

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import os
import numpy as np

# IMPORT PROJECT FILES
from dataloader import dataset
from network_utils import train, convert_image_np, evaluate, visualize_stn, visualize_scalar
from load_save import Logger, save_stn_data, save_stn_cnn, load_stn_data, load_stn_cnn
from network import CNN_STN


if __name__ == "__main__":
    filepath_this_file = os.path.dirname(os.path.abspath(__file__))
    project_path,_ = os.path.split(filepath_this_file)
    data_path = os.path.join(project_path + "/Data")
    #filepath_this_file = "/media/oliver/Gemeinsame Daten/ORIU/Project/Data"
    filepath_train = os.path.join(data_path + "/GTSRB/Final_Training/Images")
    filepath_test = os.path.join(data_path + "/GTSRB/Final_Test/Images")

    trained = False
    
    if not trained:
        
        trainset = dataset(filepath_train, split="train")
        #trainset.subset(0.1, fractional=True)
    
        testset = dataset(filepath_test, split="test")
        #testset.subset(0.1, fractional=True)
    
        print("Trainset: " + str(len(trainset)))
        print("Testset: " + str(len(testset)))
    
        dataloader_train = DataLoader(trainset, batch_size=32, shuffle=True)
        dataloader_test = DataLoader(testset, batch_size=32, shuffle=True)
        
        use_gpu = torch.cuda.is_available()
        tqdm.write("CUDA is available: " + str(use_gpu))
    
        # Model creation and training
        model = CNN_STN(43, use_gpu=use_gpu)
        #model.use_gpu=use_gpu
        logger = Logger()
        
        model.databatch=next(iter(dataloader_train))["tensor"].cuda()
        train(model, dataloader_train, n_epochs=50, checkpoint_name="test", use_gpu=use_gpu, stn=True, dataloader_test=dataloader_test, logger=logger)
    
        print("Train accuracy: " + str(evaluate(model, dataloader_train)))
        print("Test accuracy: " + str(evaluate(model, dataloader_test)))

        save_stn_data(model, "./Saved/stn_data_test")
        save_stn_cnn(model, "./Saved/stn_cnn_test")
        logger.save("./Saved/logger_test")

    else:
        trainset = dataset(filepath_train, split="train")
        trainset.subset(0.1, fractional=True)
    
        testset = dataset(filepath_test, split="test")
        testset.subset(0.1, fractional=True)
    
        print("Trainset: " + str(len(trainset)))
        print("Testset: " + str(len(testset)))
    
        dataloader_train = DataLoader(trainset, batch_size=32, shuffle=True)
        dataloader_test = DataLoader(testset, batch_size=32, shuffle=True)
        
        use_gpu = torch.cuda.is_available()
        tqdm.write("CUDA is available: " + str(use_gpu))
        
        model = CNN_STN(43, use_gpu=use_gpu)
        #model.use_gpu=use_gpu
        logger = Logger()
        
        load_stn_cnn(model, "./Saved/stn_cnn_test")
        model.list_thetas = load_stn_data("./Saved/stn_data_test")
        print(load_stn_data("./Saved/stn_data_test"))
        logger.load("./Saved/logger_test")
        
        print(model.list_thetas)
        
        print("Train accuracy: " + str(evaluate(model, dataloader_train)))
        print("Test accuracy: " + str(evaluate(model, dataloader_test)))
        
        fig = plt.figure(figsize=(8,5), dpi=100)
        
        epoch_list=[0,1,2,3,3,3]
        for i in range(6):
            fig.add_subplot(2,3,1+i)
            plt.imshow(model.list_thetas[epoch_list[i]])
            plt.xticks([]), plt.yticks([])
            plt.title('Transformed Images ' + str(epoch_list[i]) + ' epochs', fontsize=9)
    
        fig.tight_layout()

        plt.show(fig)
        fig.savefig('./Plots/stn_test.pdf', dpi=300)
    
        visualize_stn(model)
    
        visualize_scalar(logger.test_acc_list, "./Plots/test_2.pdf", "test of accuracy loading", xname="epoch", yname="acc")
        
        visualize_scalar(logger.loss_list, filename="./Plots/loss_2.pdf", title="Loss of total network", xname="batch", 
                         yname="loss", show=False, scalars=1, labels=None, ylim=(0,6))
        
        visualize_scalar(logger.batch_acc_list, filename="./Plots/batch_acc_2.pdf", title="Accuracy of batch", xname="batch", 
                         yname="accuracy", show=False, scalars=1, labels=None)
        
        visualize_scalar(model.list_train_acc, filename="./Plots/train_acc_2.pdf", title="Accuracy of Trainset", xname="epoch", 
                             yname="accuracy", show=False, scalars=1, labels=None)
        
        train = np.array(model.list_train_acc)
        test = np.array(model.list_test_acc)
        xlen = train.shape[0]
        ylen = 4
        data = np.zeros((xlen,ylen))
        data[:,0:2] = train
        data[:,2:4] = test
        labels = ["Trainset", "Testset"]
        visualize_scalar(data, filename="./Plots/train_test_acc_2.pdf", title="Accuracy of Datasets", xname="epoch", 
                             yname="accuracy", show=False, scalars=2, labels=labels)
    