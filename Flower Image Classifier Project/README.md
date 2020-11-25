# Flower Image Classifier Project

In this project, I trained an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. I'll be using a dataset of 102 flower categories to train a deep neural network and predict from.

In this project, I used PyTorch to build a deep learning neural network. Specifically, I downloaded pretrained neural nets from torchvision.models and further altered the classifier to better fit our specifi problem. I kept the features used in the pretrained network.

## Python Notebook
There is a Python Notebook detailing some initial analysis and tinkering to then further build python files that could be run on the Command Line.

## To Run:
### Train.py
To run the train.py file, we do the following on the Command Line:

*Basic usage: python train.py data_directory*

Prints out training loss, validation loss, and validation accuracy as the network trains

Options:
  * Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
  * Choose architecture: python train.py data_dir --arch "vgg13"
  * Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
  * Use GPU for training: python train.py data_dir --gpu
  
This will build a model and save it as a 'checkpoint' to be used in our predict.py file.   


### Predict.py
To run the predict.py file, we do the following on the Command Line:

*Basic usage: python predict.py /path/to/image checkpoint*

Options:
  * Return top KK most likely classes: python predict.py input checkpoint --top_k 3
  * Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
  * Use GPU for inference: python predict.py input checkpoint --gpu
