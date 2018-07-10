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
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Linear, Conv2d, ReLU, MaxPool2d, CrossEntropyLoss, SmoothL1Loss
from torch.nn.init import xavier_normal_, normal_
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms
from dataloader import *

class CNN_STN(nn.Module):
    '''
    Example from exercises
    '''
    def __init__(self, n_classes):
        '''
        Arguments: n_classes -  number of classes
        '''
        super(CNN_STN, self).__init__()
        
        # Convolutional network
        self.conv1 = Conv2d(4, 100, 7, stride=1, padding=1)
        self.conv2 = Conv2d(100, 150, 5, stride=1, padding=1)
        self.conv3 = Conv2d(150, 250, 5, stride=1, padding=2)
        
        # Maxpool
        self.maxpool = MaxPool2d(2, 2)
        
        # Fully connected network
        # input units: ( [[[(32-4)/2] -2 ]/ 2] -0)**2 * 250 = 6**2 * 250
        #               \__________________________/     \
        #             output tensor size                channels
        self.n_units = 6*6*250
        self.fc1 = Linear(self.n_units, 300)
        self.fc2 = Linear(300, n_classes)

        self.activation = ReLU()
        # Initialize all layers of CNN
        self.apply(initializer)
        
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(4,100, kernel_size=5, stride=1, padding=0),
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(100,200, kernel_size=5, stride=1, padding=1),
                nn.ReLU(True)#,
                #nn.MaxPool2d(2, stride=2)
        )
        
        # input units: ( [[((32/2)-4)/2] -2 ]/ 2)**2 * 200 = 2**2 * 200
        #               \__________________________/     \
        #             output tensor size                channels
        
        #self.n_units_stn = 2*2*200
        self.n_units_stn = 4*4*200
        
        # Regressor for the 2x3 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.n_units_stn, 200),
            nn.ReLU(True),
            nn.Linear(200,2*3)
        )
        
        #Initializer all layer of STN
        self.localization.apply(initializer_stn)
        self.fc_loc.apply(initializer_stn)
        
        # Initialize the weights/bias with identity transformation of last STN layer
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        
        # Save stuff for evaluation:
        self.databatch = None
        self.list_thetas = []
        self.theta=0
        
        return None
        
    def stn(self, x): 
        # Calculate convolutional output
        temp = self.localization(x)
        temp = temp.view(-1, self.n_units_stn)
        
        # Regress the transformation matrices
        theta = self.fc_loc(temp)
        theta = theta.view(-1,2,3)
        
        # Apply the transformation
        grid = nn.functional.affine_grid(theta, x.size())
        out = nn.functional.grid_sample(x, grid)
        
        return out, theta

    def forward(self, x, skip_stn=False):
        # Transform the input
        if not skip_stn:
            out, theta = self.stn(x)
        else:
            out = x
            theta = 0
        # Apply the CNN
        out = self.conv1(out)
        out = self.activation(out)
        out = self.maxpool(out)
        
        out = self.conv2(out)
        out = self.activation(out)
        out = self.maxpool(out)
        out = self.conv3(out)
        out = self.activation(out)
        
        out = out.view(-1, self.n_units)
        
        out = self.fc1(out)
        out = self.activation(out)

        self.features = out

        out = self.fc2(out)

        return out, theta
        
    def save_stn(self):
        with torch.no_grad():
            # Get a batch of training data
            data = self.databatch.clone()

            input_tensor = data.data
            transformed_input_tensor, theta = self.stn(input_tensor)
            out_grid = convert_image_np(torchvision.utils.make_grid(transformed_input_tensor.narrow(1,0,3)).cpu())

            self.list_thetas.append(out_grid)
        
        return None

class CNN(nn.Module):
    '''
    Example from exercises
    '''
    def __init__(self, n_classes):
        '''
        Arguments: n_classes -  number of classes
        '''
        super(CNN, self).__init__()
        
        self.conv1 = Conv2d(4, 100, 7, stride=1, padding=1)
        self.conv2 = Conv2d(100, 150, 5, stride=1, padding=1)
        self.conv3 = Conv2d(150, 250, 5, stride=1, padding=2)
        
        self.maxpool = MaxPool2d(2, 2)

        # input units: ( [[[(32-4)/2] -2 ]/ 2] -0)**2 * 250 = 6**2 * 250
        #               \__________________________/     \
        #             output tensor size                channels
        self.n_units = 6*6*250
        self.fc1 = Linear(self.n_units, 300)
        self.fc2 = Linear(300, n_classes)

        self.activation = ReLU()

        self.apply(initializer)
        
        return None

    def forward(self, x):
        out = self.conv1(x)
        out = self.activation(out)
        out = self.maxpool(out)
        
        out = self.conv2(out)
        out = self.activation(out)
        out = self.maxpool(out)

        out = self.conv3(out)
        out = self.activation(out)
        
        out = out.view(-1, self.n_units)
        
        out = self.fc1(out)
        out = self.activation(out)

        self.features = out

        out = self.fc2(out)

        return out

def convert_image_np(tensor):
    """Convert a Tensor to numpy image."""
    np_array = tensor.numpy()
    np_array = np.moveaxis(np_array,0,-1)
    np_array = np_array[...,:3]
    return np_array.astype(np.uint8)
    
def train(model, dataloader, n_epochs=10, checkpoint_name='training', use_gpu=True, stn=True):
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
    Distance = SmoothL1Loss(size_average=False)
    Optimizer = torch.optim.Adam(model.parameters())
    
    #if use_gpu:
        #model.databatch=next(iter(dataloader))["tensor"].cuda()
    #else:
        #model.databatch=next(iter(dataloader))["tensor"]
    
    # We calculate for all epochs
    for epoch in tqdm(range(n_epochs), desc='epoch', position=1):
        
        # We loop through the set of batches
        for batch_index, batch in enumerate(tqdm(dataloader, desc='batch', position=0)):
            train_step = batch_index + len(dataloader)*epoch
            
            #lr=1
            #if epoch % 1 == 0 and epoch <= 3:
            #    lr /= 10
            #Optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
                                          
                                          
            # Unpack batch
            images_batch, ids_batch = batch['tensor'], batch['id']
            
            # Transform to variabels
            
            images_batch = Variable(images_batch)
            ids_batch = Variable(ids_batch)
            #print(ids_batch)
            if use_gpu:
                images_batch = images_batch.cuda()
                ids_batch = ids_batch.cuda()

            # Forward
            if epoch < 6 :
                predictions, thetas = model(images_batch, True)
                
                # Loss
                loss2 = Loss(predictions, ids_batch)
                loss = loss2
            
            else:
                predictions, thetas = model(images_batch, False)
                N_thetas = [*thetas.size()][0]
                identity_tensor = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).repeat((N_thetas,1,1))
                if use_gpu:
                    identity_tensor = identity_tensor.cuda()
                    
                # Loss
                loss1 = Distance(thetas, identity_tensor)
                loss2 = Loss(predictions, ids_batch)
                loss = 0.01 * loss1 + loss2
                #loss = loss2
                #if epoch  < 3:
                #    loss = loss2 + loss1
                #else:
                #    loss = loss2
            
            
            
            acc = torch.mean(torch.eq(torch.argmax(predictions, dim=-1),
                                      ids_batch).float())
            
            # Zero the gradient before backward propagation
            Optimizer.zero_grad()

            # Backward propagation
            loss.backward()
            
            # Update
            Optimizer.step()
            
            if train_step % 25 == 0:
            #if batch_index == len(dataloader)-2:
                tqdm.write('{}: Batch-Accuracy = {}, Loss = {}, Epoch = {}'\
                          .format(train_step, float(acc), float(loss), epoch))
                if stn:
                    visualize_stn(model)
                # Evaluation set up: Save theta after 
                #model.save_stn()
        
        # Save the model after every epoch
        torch.save(model.state_dict(), '{}-{}.ckpt'.format(checkpoint_name, epoch))
        
        # Evaluation set up: Save theta after 
        if stn:
            model.save_stn()
    return None

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
        
def initializer_stn(module):
    if isinstance(module, Conv2d):
        xavier_normal_(module.weight, gain=0.3)
        normal_(module.bias, std=0.3)
    elif isinstance(module, Linear):
        normal_(module.weight, std=0.3)
        normal_(module.bias, std=0.3)
    


def evaluate(model, dataloader, use_gpu=True):
    """
    For a CNN model we calculate the accuracy on the dataset of dataloader.
    Arguments:  model - a CNN instance
                dataloader - a dataloader instance bases on dataset or *** class
    """
    if use_gpu:
        model.cuda()
    
    acc = 0
    for batch in tqdm(dataloader, desc="Evaluation", position=2):
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

def visualize_stn(model):
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
        fig.savefig('stn_test_2.pdf', dpi=300)
    

# Testing stuff
        
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    #filepath_this_file = os.path.dirname(os.path.abspath(__file__))
    filepath_this_file = "/media/oliver/Gemeinsame Daten/ORIU/Project/Data"
    filepath_train = os.path.join(filepath_this_file + "/GTSRB/Final_Training/Images")
    filepath_test = os.path.join(filepath_this_file + "/GTSRB/Final_Test/Images")
    
    trainset = dataset(filepath_train, split="train")
    trainset.subset(0.9, fractional=True)
    
    testset = dataset(filepath_test, split="test")
    testset.subset(0.4, fractional=True)
    
    print("Trainset: " + str(len(trainset)))
    print("Testset: " + str(len(testset)))
    
    dataloader_train = DataLoader(trainset, batch_size=64, shuffle=True)
    dataloader_test = DataLoader(testset, batch_size=64, shuffle=True)
    
    #labels=[]
    #for i in range(len(trainset)):
    #    labels.append(trainset[i]["id"])
    #print(labels)
    
    use_gpu = torch.cuda.is_available()
    tqdm.write("CUDA is available: " + str(use_gpu))
    
    #model = CNN_STN(43)
    model = CNN_STN(43)
    model.databatch=next(iter(dataloader_train))["tensor"].cuda()
    train(model, dataloader_train, n_epochs=20, checkpoint_name="test", use_gpu=True, stn=True)
    print("Train accuracy: " + str(evaluate(model, dataloader_train)))
    print("Test accuracy: " + str(evaluate(model, dataloader_test)))

    # Plot the results of STN side-by-side
    tensor = model.databatch.cpu()
    original = convert_image_np(torchvision.utils.make_grid(tensor.narrow(1,0,3)).cpu())
    
    fig = plt.figure(figsize=(8,5), dpi=100)
        
    fig.add_subplot(2,3,1)
    plt.imshow(original)
    plt.xticks([]), plt.yticks([])
    plt.title('Dataset Images', fontsize=9)
    
    epoch_list=[0,0,0,0,0]
    for i in range(1):
        fig.add_subplot(2,3,2+i)
        plt.imshow(model.list_thetas[i])
        plt.xticks([]), plt.yticks([])
        plt.title('Transformed Images ' + str(epoch_list[i]+1) + ' epochs', fontsize=9)
    
    fig.tight_layout()

    plt.show(fig)
    fig.savefig('stn_test.pdf', dpi=300)
    
    visualize_stn(model)
    
    temp = model.localization(tensor.cuda())
    temp = temp.view(-1, model.n_units_stn)
        
    # Regress the transformation matrices
    theta = model.fc_loc(temp)
    theta = theta.view(-1,2,3)
    
    print(theta)
    