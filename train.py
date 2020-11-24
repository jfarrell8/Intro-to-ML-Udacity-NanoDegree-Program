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
    if len(sys.argv) == 2:
        data_directory = sys.argv[1]
    
        # get category class mapping to names for flowers
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
        
        # capture train and validation data and transform
        train_data, valid_data = th.load_data(data_directory)
        trainloader, validloader, class_to_idx = th.transform_data(train_data, valid_data)     
        
        # Use GPU if it's available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print('Building model...')
        model = models.vgg11(pretrained=True)
                
        # update model classifier to customize for our problem (102 category probability calcs)
        model.classifier = th.update_classifier(model)
        
        print('Training model...')
        th.train_model(model, trainloader, validloader)
        
        # add class_to_idx attribute to model
        model.class_to_idx = class_to_idx
        
        print('Saving model...')
        th.save_model(model)

        print('Trained model saved!')

    else:
        print('Please provide the flowers directory path as the first argument'\
              '\n\nExample: python train.py flowers')


if __name__ == '__main__':
    main()