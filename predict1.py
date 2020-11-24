import sys
import argparse
import json
import torch
import predict_helper as ph

def main():
    parser = argparse.ArgumentParser(description='Optional add-ons.')
    parser.add_argument('image_filepath')
    parser.add_argument('checkpoint')
    parser.add_argument('--topk', action='store', type=int)
    parser.add_argument('--gpu', action='store_true')
    #parser.add_argument('--category_names', action='store') 
    args = parser.parse_args()
    
    if args.topk:
        topk = args.topk
    else:
        topk = 1
    
    if args.gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # assign model from checkpoint
    model = ph.load_checkpoint(args.checkpoint, device)

    # assign predicted top "k" probabilities and classes
    probs, classes = ph.predict(args.image_filepath, model, topk)

    # get category class mapping to names for flowers
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # convert class ID to a category name
    cat_class = cat_to_name[classes[0]]
    
    print('Flower Name: ', cat_class)
    print('Top Probability(ies): ', probs[:args.topk])

        
if __name__ == '__main__':
    main()