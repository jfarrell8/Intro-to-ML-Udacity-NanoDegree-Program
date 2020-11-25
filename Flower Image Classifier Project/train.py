import sys
import argparse
import json
import torch
from torchvision import models
import train_helper as th

# Basic usage: python train.py data_directory
# Prints out training loss, validation loss, and validation accuracy as the network trains
# Options:
# Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
# Choose architecture: python train.py data_dir --arch "vgg13"
# Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
# Use GPU for training: python train.py data_dir --gpu

def main():
    # Basic usage: python train.py data_directory
    parser = argparse.ArgumentParser(description='Optional add-ons.')
    parser.add_argument('data_directory')
    parser.add_argument('--save_dir', action='store')
    parser.add_argument('--arch', action='store')
    parser.add_argument('--learning_rate', action='store', type=float)
    parser.add_argument('--hidden_units', action='store', type=int)
    parser.add_argument('--epochs', action='store', type=int)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    
    # get category class mapping to names for flowers
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # capture train and validation data and transform
    train_data, valid_data = th.load_data(args.data_directory)
    trainloader, validloader, class_to_idx = th.transform_data(train_data, valid_data)     

    # Use GPU if it's available
    device = torch.device('cuda') if (args.gpu | torch.cuda.is_available()) else torch.device('cpu')

    print('Building model...')
    # set default to vgg11
    args.arch = args.arch if args.arch else 'vgg11'
    model = models.__dict__[args.arch](pretrained=True)
    model.arch = args.arch
    
    # update model classifier to customize for our problem (102 category probability calcs)
    hidden_units = args.hidden_units if args.hidden_units else 4098
    model.classifier = th.update_classifier(model, hidden_units)
    
    print('Training model...')
    epochs = args.epochs if args.epochs else 5
    learning_rate = args.learning_rate if args.learning_rate else 0.001
    th.train_model(model, trainloader, validloader, epochs, learning_rate, device)

    # add class_to_idx attribute to model
    model.class_to_idx = class_to_idx

    print('Saving model...')
    save_dir = args.save_dir if args.save_dir else 'saved_models'
    th.save_model(model, save_dir)

    print('Trained model saved!')


if __name__ == '__main__':
    main()