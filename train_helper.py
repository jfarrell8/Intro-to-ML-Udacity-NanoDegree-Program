from collections import OrderedDict

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

def load_data(data_directory):
    # define train, valid, and test directories
    train_dir = data_directory + '/train'
    valid_dir = data_directory + '/valid'
    
    return train_dir, valid_dir

def transform_data(train_dir, valid_dir):
    # Define your transforms for the training, validation, and testing sets
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

def update_classifier(model):
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # rebuild classifier feed forward since the VGG network doesn't fit our particular dataset
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, 4096)),
                              ('relu1', nn.ReLU()),
                              ('fc2', nn.Linear(4096, 1000)),
                              ('relu2', nn.ReLU()),
                              ('dp', nn.Dropout(0.1)),
                              ('fc3', nn.Linear(1000, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    
    return classifier

def train_model(model, trainloader, validloader):
    
    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # model to GPU if needed
    model = model.to(device)
    
    # define the loss to NLLLoss since logsoftmax used for outputs
    criterion = nn.NLLLoss()

    # set optimizer given classifier parameters; may need to alter learning_rate later
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    # train the model
    epochs = 5
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

def save_model(model):
    checkpoint = {'arch': 'vgg11',
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()
                 }

    torch.save(checkpoint, 'checkpoint.pth')