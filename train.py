#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */aipnd-project/train.py

# PROGRAMMER: LIM TERN POH                                                    
# DATE CREATED: 02/16/2018     
# PURPOSE: Develop an image classifer using PyTorch and pre-trained CNN.

# Use argparse Expected Call with <> indicating expected user input:
#   python train.py --data_dir <directory with images>  --arch <model>  --hidden_units <no. of hidden units> --learning_rate <learning rate for CNN model>  --epochs <number of epochs to run>  --gpu <whether to train on GPU>

# Imports modules
import numpy as np
import torchvision
import torch

from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from collections import OrderedDict

import argparse
from time import time

def main():
    start_time = time()
    in_arg = get_input_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() and in_arg.gpu else "cpu")

    # Load and prepare data
    test_datasets, test_dir, train_datasets, train_loader, test_loader = load_data(in_arg.data_dir)
    
    # Build classifier
    model, input_size = load_arch(in_arg.arch)
    criterion, optimiser = build_classifier(in_arg.hidden_units, in_arg.learning_rate, model, input_size, device)
    
    # Train, test, and validate classifier
    validation(test_loader, device, model, criterion)
    train_classifier(in_arg.epochs, model, optimiser, device, criterion, train_loader, test_loader)
    check_accuracy_on_test(test_loader, device, model)
    
    # Save checkpoint
    model.class_to_idx = train_datasets.class_to_idx
    torch.save(model, 'check_point.pth')
    
    # Computes overall runtime in seconds
    end_time = time()
    tot_time = end_time - start_time
    
    # Prints overall runtime in format hh:mm:ss
    print("\nTotal Elapsed Runtime:", str( int( (tot_time / 3600) ) ) + ":" + 
          str( int(  ( (tot_time % 3600) / 60 )  ) ) + ":" + 
          str( int( ( (tot_time % 3600) % 60 ) ) ) )

def get_input_args():
    """
     Retrieves and parses the command line arguments created and defined using the argparse module. This function returns these arguments as an ArgumentParser object.
     Parameters:
        None - simply using argparse module to create & store command line arguments
     Returns:
        parse_args() -data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser(description='Image Classifier')
    
    parser.add_argument('--data_dir', type=str, default='flowers', help='Path to image directory with 3 subdirectories, "train", "valid", and "test"')
    parser.add_argument('--arch', type=str, default='vgg19', help='CNN model for image classification; choose either "vgg19" or "alexnet" only')
    parser.add_argument('--hidden_units', type=int, default=4096, help='Number of hidden units')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the CNN model')
    parser.add_argument('--epochs', type=int, default=9, help='Number of epochs to run')
    parser.add_argument('--gpu', type=bool, default=True, help='Train classifier on GPU?')

    return parser.parse_args()

def load_data(data_dir):
    '''
    Load data set with torchvision's ImageFolder
    Parameters:
        data_dir: path to the image folder. Required subdirectories are "train", "valid", and "test"
    Returns:
        parse_args() - data structure that stores the CLA object
    '''
    
    #Set folder path
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Defining transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
    }
    
    #Load datasets
    train_datasets = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    valid_datasets = datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
    test_datasets = datasets.ImageFolder(test_dir, transform=data_transforms['test'])

    #Load dataloaders
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=True)
    
    return test_datasets, test_dir, train_datasets, train_loader, test_loader

def load_arch(arch):
    '''
    Load a pretrained CNN network for image classification; only "vgg19" and "alexnet" can be used
    '''
    
    if arch=='vgg19':
        model = models.vgg19(pretrained=True)
        input_size = 25088
    elif arch=='alexnet':
        model = models.alexnet(pretrained=True)
        input_size = 9216
    else:
        raise ValueError('Please choose either "vgg19" or "alexnet"')
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model, input_size

def build_classifier(hidden_units, learning_rate, model, input_size, device):
        classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout()),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
        ]))

        model.classifier = classifier    
        model.to(device)

        criterion = nn.NLLLoss()
        optimiser = optim.Adam(model.classifier.parameters(), lr=learning_rate)        

        return criterion, optimiser
    
def validation(test_loader, device, model, criterion):
    test_loss = 0
    accuracy = 0
    
    for inputs, labels in test_loader:

        inputs, labels = inputs.to(device), labels.to(device)

        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

def train_classifier(epochs, model, optimiser, device, criterion, train_loader, test_loader):
    epochs = epochs
    print_every = 64

    for e in range(epochs):
        running_loss = 0
        steps = 0

        start = time()

        model.train()
        for inputs, labels in train_loader:
            steps += 1

            optimiser.zero_grad()

            inputs, labels = inputs.to(device), labels.to(device)

            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimiser.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Set network in evaluation mode for inference
                model.eval()

                # Turn off gradients for validation to save memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(test_loader, device, model, criterion)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(test_loader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(test_loader)),
                      "Device: {}...Time: {:.3f}s".format(device, (time() - start)/3))

                running_loss = 0
                start = time()

                # Turn training back on
                model.train()
    print("End")

# Test trained network on test data 
def check_accuracy_on_test(test_loader, device, model):
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))     #ok
    
# Call to main function to run the program
if __name__ == '__main__':
    main()