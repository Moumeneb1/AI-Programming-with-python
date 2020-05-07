import argparse
from PIL import Image
from model import ImgModel
import numpy as np
import torch
import json
import matplotlib.pyplot as plt





def read_cat(path):
    with open(path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def predict(image_path, model,device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    image = torch.from_numpy(image)
    image.unsqueeze_(0)
    image = image.float()
    image = image.to(device)
    outputs = model(image)
    probs, classes = torch.exp(outputs).topk(topk)
    return probs[0].tolist(), classes[0].add(1).tolist()


def parse():
    parser = argparse.ArgumentParser(description='Let\'s test your network!')
    parser.add_argument('image_input', help='image file to classifiy (required)')
    parser.add_argument('model_checkpoint', help='model used for classification (required)')
    parser.add_argument('--top_k', type=int,help='how many prediction categories to show.',default=5)
    parser.add_argument('--category_names', help='file for category names' , default ='cat_to_name.json')
    parser.add_argument('--gpu', action='store_true', help='gpu option',default=False)
    args = parser.parse_args()
    return args

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    
    
    img = Image.open(image)    
    wpercent = (256/float(min(img.size)))
    hsize = int(float(max(img.size))*float(wpercent))
    img = img.resize((256,hsize), Image.ANTIALIAS) if img.size[0]<img.size[1] else img.resize((hsize,256), Image.ANTIALIAS)
    
    width, height = img.size
                
    left = (width - 244)/2
    top = (height - 244)/2
    right = (width + 244)/2
    bottom = (height + 244)/2
                
    img = img.crop((left, top, right, bottom))
    np_image = np.array(img)
    np_image = np_image.astype('float64')
    np_image = np_image / [255,255,255]
    np_image = (np_image - [0.485, 0.456, 0.406])/ [0.229, 0.224, 0.225]
    np_image = np_image.transpose((2, 0, 1))
    return np_image

def display_prediction(results,category_names):
    cat_file = read_cat(category_names)
    for p, c in zip(results[0],results[1]):
        c = cat_file.get(str(c),'None')
        print("{} avec {:.2f}".format(c,p))
    return None


def main():
    args = parse() 
    model = ImgModel.load(args.model_checkpoint)
    if args.gpu : 
        device = 'cuda'
    else : 
        device = 'cpu'
    model.to(device)
    prediction = predict(args.image_input,model,torch.device(device),args.top_k)
    display_prediction(prediction,args.category_names)
    
    

    
    
    
main()