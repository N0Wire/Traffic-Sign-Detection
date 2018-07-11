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
        
        # Convolutional network
        self.conv1 = Conv2d(4, 100, 7, stride=1, padding=1)
        self.conv2 = Conv2d(100, 200, 5, stride=1, padding=1)
        self.conv3 = Conv2d(200, 300, 5, stride=1, padding=2)
        
        # Maxpool
        self.maxpool = MaxPool2d(2, 2)
        
        # Fully connected network
        # input units: ( [[[(32-4)/2] -2 ]/ 2] -0)**2 * 250 = 6**2 * 300
        #               \__________________________/     \
        #             output tensor size                channels
        self.n_units = 6*6*300
        self.fc1 = Linear(self.n_units, 350)
        self.fc2 = Linear(350, n_classes)

        self.activation = ReLU()
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
        
        # input units: ( [[((32/2)-4)/2] -2 ]/ 2)**2 * 200 = 2**2 * 200
        #               \__________________________/     \
        #             output tensor size                channels
        
        #self.n_units_stn = 2*2*200
        self.n_units_stn = 4*4*300
        
        # Regressor for the 2x3 affine matrix
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
        self.theta=0
        
        return None
        
    def stn(self, x): 
        """
        The STN forward function. The CNN layers feed into the FC layers, which
        regress the two parameters for zooming and rotation of the affine
        transformation matrix:
            
            theta = ( zoom       -rotation    0  )
                    ( rotation   zoom         0  )
                    
        Arguments: input tensor
        """
        
        # Calculate convolutional output
        temp = self.localization(x)
        temp = temp.view(-1, self.n_units_stn)
        
        # Regress the transformation matrices
        theta = self.fc_loc(temp)
        theta = theta.view(-1,1,2)
        
        zoom = theta.narrow(2,0,1)
        rotation = theta.narrow(2,1,1)
        
        # We only allow for zooming and rotation in the trafo
        N_thetas = [*theta.size()][0]
        identity_tensor = Variable(torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]).repeat((N_thetas,1,1)), requires_grad=False)
        rotation_tensor = Variable(torch.tensor([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]]).repeat((N_thetas,1,1)), requires_grad=False)
        if use_gpu:
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

def convert_image_np(tensor):
    """
    Convert a Tensor with arbitrary channel number (e.g. 4 with canny edge) to numpy image.
    Arguments:  tensor - tensor of image or batch of images
    """
    np_array = tensor.numpy()
    np_array = np.moveaxis(np_array,0,-1)
    np_array = np_array[...,:3]
    return np_array.astype(np.uint8)
    
def train(model, dataloader, n_epochs=10, checkpoint_name='training', use_gpu=True, stn=True):
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
    Loss = CrossEntropyLoss()
    Distance = SmoothL1Loss(size_average=False)
    Optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.1)
    
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

            # Forward
            # We pretrain the CNN classifier without STN for 6 epochs
            if epoch < 6 :
                predictions, thetas = model(images_batch, skip_stn=True)
                
                # Loss without regulator term
                loss2 = Loss(predictions, ids_batch)
                loss = loss2
            
            else:
                predictions, thetas = model(images_batch, skip_stn=False)
                
                # Build identity tensor for L1 distance
                N_thetas = [*thetas.size()][0]
                identity_tensor = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).repeat((N_thetas,1,1))
                if use_gpu:
                    identity_tensor = identity_tensor.cuda()
                    
                # Construct losses
                loss1 = Distance(thetas, identity_tensor)
                loss2 = Loss(predictions, ids_batch)
                
                # We push the transformation close to the identity by using
                # a regulator for one epoch
                if epoch  < 7 and epoch >= 6:
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
            
            # Write the current batch accuracy and show the current trafo
            if train_step % 50 == 0:
                tqdm.write('{}: Batch-Accuracy = {}, Loss = {}, Epoch = {}'\
                          .format(train_step, float(acc), float(loss), epoch))
                if stn:
                    visualize_stn(model)
        
        # Save the model after every fourth epoch
        if epoch %4 == 0:
            torch.save(model.state_dict(), '{}-{}.ckpt'.format(checkpoint_name, epoch))
        
        # Evaluation set up: Save theta after all epochs 
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
    """
    Initialize the weights with smaller std. deviations for the STN layers
    """
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
                use_gpu - (boolean) True if we use cuda
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

def visualize_stn(model, filename="stn_test_2.pdf"):
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
    model.databatch=next(iter(dataloader_train))["tensor"].cuda()
    train(model, dataloader_train, n_epochs=30, checkpoint_name="test", use_gpu=use_gpu, stn=True)
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
    