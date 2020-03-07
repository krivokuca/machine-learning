import numpy as np
import torch.nn as nn
import torch.cuda
import torch.optim as optim
import torch.nn.functional as F
import torch.utils as utils
from time import time
from torchvision import datasets, transforms


class MNISTNetwork(nn.Module):
    '''
    Very Simple MNIST example using Pytorch and it's dataloaders
    @author Daniel Krivokuca
    @date 2020-03-07
    '''

    def __init__(self):
        '''
        Initializes our model
        '''
        super(MNISTNetwork, self).__init__()  # pytorch boilerplate
        # these correspond to the size of our input neurons, layer 1 and layer 2 neurons
        # and our output neuron
        self.SIZES = {
            'input': 784,
            'layer1': 128,
            'layer2': 64,
            'output': 10
        }

        self.has_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda")

        # our layers are here
        self.input_layer = nn.Linear(
            self.SIZES['input'], self.SIZES['layer1']).to(self.device)
        self.layer_1 = nn.Linear(
            self.SIZES['layer1'], self.SIZES['layer2']).to(self.device)
        self.output = nn.Linear(
            self.SIZES['layer2'], self.SIZES['output']).to(self.device)

        # to define our model, we tell pytorch to sequentially input data from the input_layer --> layer_1 --> output
        # we also define what type of activation function we want. For the first two layers we're going to use a standard
        # rectified linear unit but for the last one we use a logarithmic sigmoid (or softmax) function. I don't really
        # know why but the paper says to do it so we're doing it
        self.model = nn.Sequential(self.input_layer, nn.ReLU(
        ), self.layer_1, nn.ReLU(), self.output, nn.LogSoftmax(dim=1)).to(self.device)

    def dataset(self):
        '''
        This function retrieves the MNIST dataset and applies a transformation + normalization
        function to it. The shape of the dataset is:
        ([batch_size, 1, 28, 28]) - images (each image has 1 colour channel an is 28x28 pixels)
        ([64]) - the label only as
        @returns : the loaded training set and value set
        '''
        # Our transformation function takes the mnist image and transforms it into a pytorch tensor.
        # it then normalizes this tensor by finding the mean of each tensor member (M1, M2,...,Mn) and
        # the standard deviation of each tensor member (S1, S2,...,Sn) for n channels. It then does
        # the following normalization function to each channel input:
        # input[channel] = (input[channel] - mean[channel]) / std[channel]
        # I don't know why we have to normalize but the paper im reading says to do it so idk
        transformation = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        training_set = datasets.MNIST(
            './data/', download=True, train=True, transform=transformation)
        value_set = datasets.MNIST(
            './data/', download=True, train=False, transform=transformation)
        # batch_size = how many samples per batch we want to load
        # shuffle = if true, data is shuffled around every epoch
        training_loader = utils.data.DataLoader(
            training_set, batch_size=128, shuffle=True)
        value_loader = utils.data.DataLoader(
            value_set, batch_size=128, shuffle=True)
        return training_loader, value_loader

    def train(self):
        '''
        Main training method of the model
        '''
        # lets define our standard gradient descent function as well as some parameters
        parameters = {
            'learning_rate': 0.001,
            'momentum': 0.8,
            'epochs': 20
        }
        print("Starting training with parameters:")
        print("lrate: {}\nmomentum: {}\nepochs: {}\n===================\n".format(
            parameters['learning_rate'], parameters['momentum'], parameters['epochs']))
        optimizer = optim.SGD(self.model.parameters(
        ), parameters['learning_rate'], parameters['momentum'])
        # load our data into memory
        training_loader, value_loader = self.dataset()
        # timing how long training takes
        start_time = time()

        # main training loop
        for epoch in range(parameters['epochs']):
            running_loss = 0
            for img, label in training_loader:
                log_loss = nn.NLLLoss()

                # flatter our image into a single 784 member long tensor so we avoid a size mismatch
                img = img.view(img.shape[0], -1)

                # send the image and label to the gpu
                img = img.to(self.device)
                label = label.to(self.device)

                # since this is our first training pass, we need to clear the previous
                # gradient value then set it again once we compute the loss between our
                # image and our label
                optimizer.zero_grad()

                # next we can feed our image into our model, getting a hypothesis of what our
                # model thinks the value is
                hypothesis = self.model(img)

                # lets see how wrong our hypothesis is and then do backpropagation given our loss
                # value
                loss = log_loss(hypothesis, label)

                loss.backward()

                # then we can recalculate our weights and biases here
                optimizer.step()

                # and compute our running loss for the current epoch
                running_loss += loss.item()
            # to find the average loss of this epoch, simply divide the sum of all the losses by the
            # number of training examples
            training_loss = running_loss/len(training_loader)
            print("----------\nEpoch: {}\nLoss {:.4f}".format(epoch, training_loss))

        # all of our epochs have finished, lets see how long it took
        end_time = time()
        # divide by 60 to get the number of mins
        total_time = (end_time - start_time / 60)
