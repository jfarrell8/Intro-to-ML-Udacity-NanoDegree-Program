from collections import OrderedDict

import time

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

def load_data(data_directory):
    """
    Load the data directory containing the training and validation data
    
    Arguments:
        data_directory: filepath for training and validation data
    
    Returns:
        train_dir: training data filepath
        valid_dir: validation data filepath
    """
    # define train, valid, and test directories
    train_dir = data_directory + '/train'
    valid_dir = data_directory + '/valid'
    
    return train_dir, valid_dir

def transform_data(train_dir, valid_dir):
    """
    Transform the training and validation image data.
    Load transformed images into an iterative dataset that our model can loop through.
    
    Arguments:
        train_dir: filepath of the training data
        valid_dir: filepath of the validation data
    
    Returns:
        trainloader: training data that will be used to train our model
        validloader: validation data that will be used to train our model
        class_to_idx: mapping of the flower classes in the training dataset to indices
    """
    # Define your transforms for the training and validation sets
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])
                                         ])

    valid_transform = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])
                                         ])

    # Load the datasets with ImageFolder
    # Pass transforms in here, then run the next cell to see how the transforms look
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transform)
    
    # A class mapping to indices from the training dataset
    class_to_idx = train_dataset.class_to_idx

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)

    return trainloader, validloader, class_to_idx

def update_classifier(model, hidden_layers):
    """
    Updates the pretrained classifier from the torch library models.
    
    Arguments:
        model: pretrained PyTorch model to update the classifier on
        hidden_layers: the number of hidden layers to input into the classifier
        
    Returns:
        classifier: updated classifier for the pretrained network
    """
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # rebuild classifier feed forward since the pretrained network doesn't fit our particular dataset
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, hidden_layers)),
                              ('relu1', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_layers, 1000)),
                              ('relu2', nn.ReLU()),
                              ('dp', nn.Dropout(0.1)),
                              ('fc3', nn.Linear(1000, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    
    return classifier

def train_model(model, trainloader, validloader, epochs, learning_rate, device):
    """
    Trains a deep learning neural network model.
    
    Arguments:
        model: model to be trained
        trainloader: images and classification data to be used for training
        validloader: images and classification data to be used for validation
        epochs (int): number of times to iterate with GD through the NN
        learning_rate (int): incremental rate to add to GD step
        device: 'cuda' if running on GPU or 'cpu' if running on CPU
    """
    
    # model to GPU (if needed)
    model = model.to(device)
    
    # define the loss to NLLLoss since logsoftmax used for outputs
    criterion = nn.NLLLoss()

    # set optimizer given classifier parameters; may need to alter learning_rate later
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # train the model
    epochs = epochs
    running_loss = 0

    for epoch in range(epochs):

        # loop through training dataset
        for inputs, labels in trainloader:
            
            # move inputs, labels to default device
            inputs, labels = inputs.to(device), labels.to(device)

            # set gradients to zero (so they don't keep accumulating)
            optimizer.zero_grad()

            # get log propabilities from forward pass
            logps = model.forward(inputs)

            # calculate the loss
            loss = criterion(logps, labels)

            # backpropogation
            loss.backward()

            # take a step with optimizer/update the model weights
            optimizer.step()

            # add loss to running_loss for calc of total loss
            running_loss += loss.item()

        # loop through validation dataset    
        else:
            valid_loss = 0
            accuracy = 0

            # set model into eval mode to eliminate backprop
            model.eval()

            # don't calculate gradients to speed this step up
            with torch.no_grad():
                for images, labels in validloader:

                    # images and labels need to be on GPU or CPU
                    images, labels = images.to(device), labels.to(device)

                    # calc log probabilities and validation loss
                    log_ps = model(images)
                    batch_loss = criterion(log_ps, labels)
                    valid_loss += batch_loss.item()

                    # calculate probabilities from log probabilities
                    ps = torch.exp(log_ps)

                    # capture top probabilities and classes of the flower to compare for accuracy calc
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            # print out current epoch with the train & valid loss plus valid accuracy   
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/len(trainloader):.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")

            # reset running_loss
            running_loss = 0

            # put model back into train mode
            model.train()

def save_model(model, save_dir):
    """
    Saves model checkpoint
    
    Arguments:
        model: trained model to be saved
        save_dir: directory to save the model
    """
    
    checkpoint = {'arch': model.arch,
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()
                 }
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    torch.save(checkpoint, save_dir + '/checkpoint' + timestamp +'.pth')