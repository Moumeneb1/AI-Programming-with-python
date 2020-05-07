from model import ImgModel
import argparse
from utils import train,test
import torch
import json
from torchvision import datasets, transforms, models
import os


def get_data(data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([
         transforms.Resize(255),
         transforms.CenterCrop(224),
         transforms.ToTensor(),                     
         transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ])

    test_transforms = transforms.Compose([
         transforms.Resize(255),
         transforms.CenterCrop(224),
         transforms.ToTensor(),                     
         transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ])

    # Pass transforms in here, then run the next cell to see how the transforms look
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    data = {'train':trainloader,'valid':validloader,'test':testloader,'labels':cat_to_name}
    
    return data



def parse():
    parser = argparse.ArgumentParser(description='Let\'s Train an Img classifier !')
    parser.add_argument('data_directory', help='data directory (required)')
    parser.add_argument('--save_dir', help='directory to save a neural network.',default='trained')
    parser.add_argument('--arch', help='models to use',default='densenet')
    parser.add_argument('--learning_rate', help='learning rate',type=float,default=0.001)
    parser.add_argument('--hidden_units',type=int, help='number of hidden units',default=4096)
    parser.add_argument('--epochs',type=int, help='epochs',default=3)
    parser.add_argument('--gpu',action='store_true', help='gpu' , default = False)
    args = parser.parse_args()
    return args


def main():
    print("Let's train your model !")
    args = parse()
    model = ImgModel(args.arch,args.hidden_units)
    if args.gpu : 
        device = 'cuda'
    else : 
        device = 'cpu'
    model.to(device)
    data=get_data(args.data_directory)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate) 
    train(model,data['train'],data['valid'],criterion,optimizer,epochs=args.epochs,device =torch.device(device))
    if not (os.path.isdir(args.save_dir)): 
        os.mkdir(args.save_dir)
    path = args.save_dir+'/model.pth'    
    model.save(path)
    print("model finished!")
    return None


main()
