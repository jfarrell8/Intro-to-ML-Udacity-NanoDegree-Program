import torch, torchvision
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

def load_checkpoint(filepath, device):
    """
    Loads a trained neural network model checkpoint.
    
    Arguments:
        filepath: filepath where saved model can be found
        device(str): variable dictating 'cpu' or 'cuda' to push model to
    
    Return:
        model: a trained model that can be used for prediction
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.to(device)

    return model

def process_image(image):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model.
    
    Arguments:
        image: the filepath where the image is located
        
    Returns:
        np_image: a Numpy array of the modified image ready for prediction
    '''
    # Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    
    # resize to 256 shortest dimension maintaining aspect ratio
    img.thumbnail((256, 256))
    
    # extract width and height from thumbnail and use to calculate center crop dimensions
    width, height = img.size
    left = (width - 224)/2
    upper = (height - 224)/2
    right = left + 224
    lower = upper + 224
        
    # crop out center to 224 x 224
    crop_img = img.crop((left, upper, right, lower))
    
    # convert cropped image to numpy array and "normalize" pixel values
    np_image = np.array(crop_img)/ 255.
    
    # normalize numpy array image using provided means and standard deviations
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # reorder image numpy array so color channel comes first
    np_image_T = np_image.transpose((2, 0, 1))
    
    # send numpy array to torch tensor
    np_image = torch.from_numpy(np_image_T)
    
    return np_image

def predict(image_path, model, topk):
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.
    
    Arguments:
        image_path: filepath for the image to predict
        model: trained model used for prediction
        topk (int): the top "k" probabilities and classes to predict
        
    Returns:
        probs: List of the top "k" probabilities for the predicted flower
        classes: List of the top "k" classes for the predicted flower.
    '''
    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
      
    # get a torch tensor image from image_path
    img = process_image(image_path)
    # add a batch size of 1 at the beginning of the torch tensor (e.g. [1, 3, 224, 224])
    img = img.unsqueeze(0)
        
    # set model into eval mode to avoid backprop
    model.eval()
    
    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model(img.float().to(device))

    ps = torch.exp(output)
    
    # capture the top k probabilities and classes
    top_p, top_class = ps.topk(topk, dim=1)
    
    # inverse mapping of idx to class
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    
    # convert torch tensor probabilities and classes to list of probs/classes
    probs = top_p.cpu().numpy()[0]
    classes = [idx_to_class[x] for x in top_class.cpu().numpy()[0]]
    
    return probs, classes