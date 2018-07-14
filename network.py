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
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn as nn
from torch.nn import Linear, Conv2d, ReLU, MaxPool2d
from torch.nn.init import xavier_normal_, normal_
from torch.utils.data import DataLoader
import torchvision

# IMPORT PROJECT FILES
from dataloader import dataset
from network_utils import train, convert_image_np, evaluate, visualize_stn
from load_save import Logger, save_stn_data, save_stn_cnn, load_stn_data, load_stn_cnn

class CNN_STN(nn.Module):
    '''
    Convolutional Neural Network (CNN) with spatial transformer network (STN).
    The CNN determines the classlabels based on supervised training with ground
    truth data. The STN is trained unsupervisedly front to end by optimizing
    the overall classification loss.
    '''
    
    # Constructor
    def __init__(self, n_classes, use_gpu=True):
        '''
        Arguments: n_classes -  number of classes
        The use_gpu boolean flag has to be set if no cuda is available.
        '''
        super(CNN_STN, self).__init__()
        
        self.use_gpu=use_gpu
        
        # Convolutional Classification Network
        self.classification = nn.Sequential(
                nn.Conv2d(4, 100, 7, stride=1, padding=1),
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(100, 200, 5, stride=1, padding=1),
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=2),
                Conv2d(200, 300, 5, stride=1, padding=2),
                nn.ReLU(True)
                )
        
        # Fully connected network
        # input units: ( [[[(32-4)/2] -2 ]/ 2] -0)**2 * 300 = 6**2 * 300
        #               \__________________________/     \
        #                output tensor size           channels
        
        self.n_units = 6*6*300
    
        
        # Fully Connected Network to output the classification
        self.fc_class = nn.Sequential(
                nn.Linear(self.n_units, 350),
                nn.ReLU(True),
                nn.Linear(350, n_classes)
                )
        
        # Convolutional network
        #self.conv1 = Conv2d(4, 100, 7, stride=1, padding=1)
        #self.conv2 = Conv2d(100, 200, 5, stride=1, padding=1)
        #self.conv3 = Conv2d(200, 300, 5, stride=1, padding=2)
        
        # Maxpool
        #self.maxpool = MaxPool2d(2, 2)
        
        # Fully connected network
        # input units: ( [[[(32-4)/2] -2 ]/ 2] -0)**2 * 300 = 6**2 * 300
        #               \__________________________/     \
        #                output tensor size           channels
        #self.n_units = 6*6*300
        #self.fc1 = Linear(self.n_units, 350)
        #self.fc2 = Linear(350, n_classes)

        #self.activation = ReLU()
        
        # Initialize all layers of CNN
        self.apply(initializer)
        
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(4,150, kernel_size=5, stride=1, padding=0),
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(150,300, kernel_size=5, stride=1, padding=1),
                nn.ReLU(True)#,
                #nn.MaxPool2d(2, stride=2)
        )
        
        # input units: ( [[((32/2)-4)/2] -2 ] )**2 * 300 = 4**4 * 300
        #               \_______________________/     \
        #             output tensor size           channels
        
        #self.n_units_stn = 2*2*200
        self.n_units_stn = 4*4*300
        
        # Regressor for the parameters of 2x3 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.n_units_stn, 350),
            nn.ReLU(True),
            nn.Linear(350,2)
        )
        
        #Initializer all layer of STN
        self.localization.apply(initializer_stn)
        self.fc_loc.apply(initializer_stn)
        
        # Initialize the weights/bias with identity transformation of last STN layer
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([0.8,0.1], dtype=torch.float))
        
        # Save stuff for evaluation:
        self.databatch = None
        self.list_thetas = []
        self.list_train_acc = []
        self.list_test_acc = []
        
        return None
        
    def stn(self, x): 
        """
        The STN forward function. The CNN layers feed into the FC layers, which
        regress the two parameters for zooming and rotation of the affine
        transformation matrix (with simplified representation):
            
            theta = ( zoom       -rotation    0  ) =  ( r*cos(phi)  -r*sin(phi)  0 )
                    ( rotation   zoom         0  )    ( r*sin(phi)   r*cos(phi)  0 )  
                    
        Arguments: input tensor
        """
        
        # Calculate convolutional output
        temp = self.localization(x)
        temp = temp.view(-1, self.n_units_stn)
        
        # Regress the transformation matrices
        theta = self.fc_loc(temp)
        theta = theta.view(-1,1,2)
        
        # Select columns with zoom and rotation paramter per image in batch
        zoom = theta.narrow(2,0,1)
        rotation = theta.narrow(2,1,1)
        
        # We only allow for zooming and rotation in the trafo
        # Calculate the transformation matrix based on the two-parameter output
        N_thetas = [*theta.size()][0]
        identity_tensor = Variable(torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]).repeat((N_thetas,1,1)), requires_grad=False)
        rotation_tensor = Variable(torch.tensor([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]]).repeat((N_thetas,1,1)), requires_grad=False)
        if self.use_gpu:
            identity_tensor = identity_tensor.cuda()
            rotation_tensor = rotation_tensor.cuda()    

        theta = zoom*identity_tensor + rotation*rotation_tensor
        
        # Apply the transformation
        grid = nn.functional.affine_grid(theta, x.size())
        out = nn.functional.grid_sample(x, grid)
        
        return out, theta

    def forward(self, x, skip_stn=False):
        """
        The forward function of the entire network. If the STN is skipped the
        image is fed directly into the classification CNN.
        
        Arguments:  x - input tensor
                    skip_stn - (boolean) Skip the STN and feed directly into
                               classifier CNN
        """
        # Transform the input
        if not skip_stn:
            out, theta = self.stn(x)
        else:
            out = x
            theta = 0
       
        # Apply the CNN
        out = self.classification(out)
        out = out.view(-1, self.n_units)
        out = self.fc_class(out)
        
        return out, theta
        
    def save_stn(self):
        """
        This function is used to save the transformed images of a sample batch.
        It can be used at any point of training and saves the 3 color channels of
        the image as numpy arrays into self.list_thetas.
        """
        with torch.no_grad():
            # Get a batch of training data
            data = self.databatch.clone()

            input_tensor = data.data
            transformed_input_tensor, theta = self.stn(input_tensor)
            out_grid = convert_image_np(torchvision.utils.make_grid(transformed_input_tensor.narrow(1,0,3)).cpu())

            self.list_thetas.append(out_grid)
        
        return None
    
    def save_acc(self, dataloader, x_value, split="train"):
        """
        Calculate the accuracy on the testset given with DataLoader object dataloader
        Arguments:  dataloader - DataLoader object containing the dataset
                    x_value - corresponding x_value to be written to list
                    split - train or test
        """
        accuracy = evaluate(self, dataloader, use_gpu=self.use_gpu)
        if split == "train":
            self.list_train_acc.append([x_value, accuracy])
        else:
            self.list_test_acc.append([x_value, accuracy])
        
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
        self.conv2 = Conv2d(100, 200, 5, stride=1, padding=1)
        self.conv3 = Conv2d(150, 300, 5, stride=1, padding=2)
        
        self.maxpool = MaxPool2d(2, 2)

        # input units: ( [[[(32-4)/2] -2 ]/ 2] -0)**2 * 250 = 6**2 * 300
        #               \__________________________/     \
        #             output tensor size                channels
        self.n_units = 6*6*300
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
    
    return None
    
def initializer_stn(module):
    """
    Initialize the weights with smaller std. deviations for the STN layers
    """
    if isinstance(module, Conv2d):
        xavier_normal_(module.weight, gain=0.3)
        normal_(module.bias, std=0.3)
    elif isinstance(module, Linear):
        normal_(module.weight, std=0.3)
        normal_(module.bias, std=0.3)
    
    return None

# Testing stuff
        
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import os
    #filepath_this_file = os.path.dirname(os.path.abspath(__file__))
    filepath_this_file = "/media/oliver/Gemeinsame Daten/ORIU/Project/Data"
    filepath_train = os.path.join(filepath_this_file + "/GTSRB/Final_Training/Images")
    filepath_test = os.path.join(filepath_this_file + "/GTSRB/Final_Test/Images")
    
    trainset = dataset(filepath_train, split="train")
    #trainset.subset(0.9, fractional=True)
    
    testset = dataset(filepath_test, split="test")
    #testset.subset(0.4, fractional=True)
    
    print("Trainset: " + str(len(trainset)))
    print("Testset: " + str(len(testset)))
    
    dataloader_train = DataLoader(trainset, batch_size=32, shuffle=True)
    dataloader_test = DataLoader(testset, batch_size=32, shuffle=True)
    
    #labels=[]
    #for i in range(len(trainset)):
    #    labels.append(trainset[i]["id"])
    #print(labels)
    
    use_gpu = torch.cuda.is_available()
    tqdm.write("CUDA is available: " + str(use_gpu))
    
    # Model creation and training
    model = CNN_STN(43, use_gpu=use_gpu)
    #model.use_gpu=use_gpu
    logger = Logger()
    model.databatch=next(iter(dataloader_train))["tensor"].cuda()
    train(model, dataloader_train, n_epochs=30, checkpoint_name="test", use_gpu=use_gpu, stn=True, dataloader_test=dataloader_test, logger=logger)
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
    
    epoch_list=[6,9,12,18,29]
    for i in range(5):
        fig.add_subplot(2,3,2+i)
        plt.imshow(model.list_thetas[epoch_list[i]])
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
    theta = theta.view(-1,1,2)
        
    zoom = theta.narrow(2,0,1)
    rotation = theta.narrow(2,1,1)
        
    #print(theta)
    # We only allow for zooming in the trafo
    N_thetas = [*theta.size()][0]
    identity_tensor = Variable(torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]).repeat((N_thetas,1,1)), requires_grad=False)
    rotation_tensor = Variable(torch.tensor([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]]).repeat((N_thetas,1,1)), requires_grad=False)
    if use_gpu:
        identity_tensor = identity_tensor.cuda()
        rotation_tensor = rotation_tensor.cuda()    
        
    theta = zoom*identity_tensor + rotation*rotation_tensor    
    
    print(theta)
    
    logger.save("logger")
    save_stn_data(model, "stn_data")
    save_stn_cnn(model, "stn_cnn")
    
    logger.load("logger")
    load_stn_data("stn_data")
    load_stn_cnn(model, "stn_cnn")