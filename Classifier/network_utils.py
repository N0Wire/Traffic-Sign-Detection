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
from torch.autograd import Variable
from tqdm import tqdm
from torch.nn import CrossEntropyLoss, SmoothL1Loss
import torchvision
from matplotlib import pyplot as plt

# IMPORT PROJECT FILES
from load_save import save_stn_cnn
       
 
def train(model, dataloader, n_epochs=10, checkpoint_name='training', use_gpu=True, stn=True, dataloader_test=None, logger=None):
    '''
    This function trains the CNN (+STN) model. The training is done for the dataset in the dataloader instance
    for multiple epochs. After every epoch the model is saved.
    
    Arguments:  model - CNN instance
                dataloader - a DataLoader instance based on a dataset or *** instance
                n_epochs - number of epochs to be trained
                checkpoint_name - Name to be specified in the saved model
                use_gpu - Boolean stating whether CUDA shall be used (check first!)
                stn - (boolean) True if model is a CNN_STN instance
    '''

    if use_gpu:
        model.cuda()
    
    """
    We use CrossEntropyLoss for the classification task.
    To push the STN transformation close towards identity we use the SmoothL1Loss
    for determining the distance of trafo to identity.
    The optimizer we use is Adam with weight_decay to push the weights towards 0
    and reduce the number of unnecessary parameters (analogous to penalty term in Lossfunction)
    """
    # Set up optimizer (Adam) and the Loss functions
    Loss = CrossEntropyLoss()
    Distance = SmoothL1Loss(size_average=False)
    Optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.1)
    
    # Set up lists for plotting
    loss_list = []
    batch_acc_list = []
    
    # We calculate for all epochs
    for epoch in tqdm(range(n_epochs), desc='epoch', position=1):
        
        # We loop through the set of batches
        for batch_index, batch in enumerate(tqdm(dataloader, desc='batch', position=0)):
            train_step = batch_index + len(dataloader)*epoch
            
            #if epoch == 6:
            #    lr = 0.01
            #    Optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.5, nesterov=True, weight_decay=0.01)
            #elif epoch == 9:
            #    lr = 0.001
            #    Optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.05, nesterov=True, weight_decay=0.01)
            #elif epoch == 40:
            #    lr = 0.0005
            #    Optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.005, nesterov=True, weight_decay=0.01)
                                          
                                          
            # Unpack batch
            images_batch, ids_batch = batch['tensor'], batch['id']
            
            # Transform to variabels
            images_batch = Variable(images_batch)
            ids_batch = Variable(ids_batch)
            
            if use_gpu:
                images_batch = images_batch.cuda()
                ids_batch = ids_batch.cuda()

            ####### Forward #######
            # We pretrain the CNN classifier without STN for 6 epochs
            if epoch < 6 :
                predictions, thetas = model(images_batch, skip_stn=True)
                
                # Loss without regulator term
                loss2 = Loss(predictions, ids_batch)
                loss = loss2
            
            else:
                predictions, thetas = model(images_batch, skip_stn=False)
                
                # Build identity tensor for L1 distance
                #N_thetas = [*thetas.size()][0]
                N_thetas = list(thetas.shape)[0]
                identity_tensor = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).repeat((N_thetas,1,1))
                if use_gpu:
                    identity_tensor = identity_tensor.cuda()
                    
                # Construct losses
                loss1 = Distance(thetas, identity_tensor)
                loss2 = Loss(predictions, ids_batch)
                
                # We push the transformation close to the identity by using
                # a regulator for one epoch
                if epoch  == 6:
                    loss = loss2 + loss1
                
                # After that we use the classification loss (which is a convex fct.)
                # to optimize all parameters (including STN)
                else:
                    loss = loss2
            
            # Calculate Batch accuracies
            acc = torch.mean(torch.eq(torch.argmax(predictions, dim=-1),
                                      ids_batch).float())
            
            # Zero the gradient before backward propagation
            Optimizer.zero_grad()

            # Backward propagation
            loss.backward()
            
            # Update
            Optimizer.step()
            
            # Set batch accuracy and loss
            if epoch >0 :
                loss_list.append([train_step, loss])
                batch_acc_list.append([train_step, acc])
            
            # Write the current batch accuracy and show the current trafo
            if train_step % 50 == 0:
                tqdm.write('{}: Batch-Accuracy = {}, Loss = {}, Epoch = {}'\
                          .format(train_step, float(acc), float(loss), epoch))
                if stn:
                    visualize_stn(model)
        
        # Save the model after every fourth epoch
        if epoch %4 == 0 and epoch > 0:
            #save_stn_cnn(model, './Temp/{}-{}'.format(checkpoint_name, epoch))
            torch.save(model.state_dict(), '{}-{}.ckpt'.format(checkpoint_name, epoch))
        
        # Evaluation set up: Save theta after all epochs 
        if stn:
            model.save_stn()
        # Save the trainset accuracy after every epooch
        model.save_acc(dataloader, epoch, split="train")
        if dataloader_test is not None:
            model.save_acc(dataloader_test, epoch, split="test")

        ###### Update the plots of loss, batch accuracy, test and train accuracy #####
        if epoch > 0:
            visualize_scalar(loss_list, filename="./Plots/loss.pdf", title="Loss of total network", xname="batch", 
                         yname="loss", show=False, scalars=1, labels=None, ylim=(0,6))
        
            visualize_scalar(batch_acc_list, filename="./Plots/batch_acc.pdf", title="Accuracy of batch", xname="batch", 
                         yname="accuracy", show=False, scalars=1, labels=None)
        
            if dataloader_test is None:
                visualize_scalar(model.list_train_acc, filename="./Plots/train_acc.pdf", title="Accuracy of Trainset", xname="epoch", 
                             yname="accuracy", show=False, scalars=1, labels=None)
            else:
                # Build data array with train and test data
                train = np.array(model.list_train_acc)
                test = np.array(model.list_test_acc)
                xlen = train.shape[0]
                ylen = 4
                data = np.zeros((xlen,ylen))
                data[:,0:2] = train
                data[:,2:4] = test
                labels = ["Trainset", "Testset"]
                visualize_scalar(data, filename="./Plots/train_test_acc.pdf", title="Accuracy of Datasets", xname="epoch", 
                             yname="accuracy", show=False, scalars=2, labels=labels)
    
    # Save all scalar values to logger for later use
    if logger is not None:
        logger.loss_list = loss_list
        logger.batch_acc_list = batch_acc_list        
        logger.test_acc_list = model.list_test_acc
        if dataloader_test is not None:
            logger.train_acc_list = model.list_train_acc
    
    return None
    

def evaluate(model, dataloader, use_gpu=True):
    """
    For a CNN model we calculate the accuracy on the dataset of dataloader.
    Arguments:  model - a CNN instance
                dataloader - a dataloader instance bases on dataset or *** class
                use_gpu - (boolean) True if we use cuda
    """
    if use_gpu:
        model.cuda()
    
    acc = 0
    for batch in tqdm(dataloader, desc="Evaluation"):
        if use_gpu:
            images_batch = batch["tensor"].cuda()
            ground_truth = batch["id"].cuda()
        else:
            images_batch = batch["tensor"]
            ground_truth = batch["id"]
        predictions, thetas = model(images_batch)
        acc += torch.mean(torch.eq(torch.argmax(predictions, dim=-1),
                                   ground_truth).float())

    acc /= len(dataloader)

    acc = float(acc.detach().cpu().numpy())

    return acc


def visualize_stn(model, filename="./Plots/stn_test_2.pdf"):
    """
    For the sample databatch saved in the model we visualize the current STN
    transformation and save it to a file.
    Arguments:  model - a CNN_STN model instance
                filename - string of the filename where to save
    """
    with torch.no_grad():
        # Get a batch of training data
        input_tensor = model.databatch
        transformed_input_tensor, thetas = model.stn(input_tensor)

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor.narrow(1,0,3)).cpu())

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor.narrow(1,0,3)).cpu())

        # Plot the results of STN side-by-side
        fig = plt.figure(figsize=(8,4), dpi=100)
        
        fig.add_subplot(1,2,1)
        plt.imshow(in_grid)
        plt.xticks([]), plt.yticks([])
        plt.title('Dataset Images', fontsize=9)
    
        fig.add_subplot(1,2,2)
        plt.imshow(out_grid)
        plt.xticks([]), plt.yticks([])
        plt.title('Transformed Images', fontsize=9)
    
        fig.tight_layout()

        plt.show(fig)
        fig.savefig(filename, dpi=300)
    
    return None

def visualize_scalar(datalist, filename, title, xname, yname, show=False, scalars=1, labels=None, ylim=None):
    """
    Plots a list of scalar values.
    Arguments:  datalist - data to be plotted in 0-column x and 1-column y-values
                filename - filename where to save the plot
                title - title of plot
                xname - label x-axis
                yname - label y-axis
                show - (boolean) show the plot in console if True
                scalars - how many scalars are supposed to be plotted into plot
                labels - list of labels for data in legend
    """
    datalist = np.array(datalist)
    
    fig_temp = plt.figure(dpi=100)
        
    fig_temp.add_subplot(1,1,1)
    for i in range(scalars):
        plt.plot(datalist[:,2*i+0], datalist[:,2*i+1])
    if labels is not None:
        plt.legend(labels)
    plt.xlabel(xname)
    plt.ylabel(yname)
    if ylim is not None:
        plt.ylim(ylim)
    plt.title(title, fontsize=9)
    
    fig_temp.tight_layout()
    
    if show:
        plt.show(fig_temp)
    fig_temp.savefig(filename, dpi=300)
    
    plt.close(fig_temp)
    
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
    