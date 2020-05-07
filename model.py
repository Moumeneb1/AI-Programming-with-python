
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models



class ImgModel(nn.Module):

    def __init__(self, arch='vgg', hidden_units='4096'):
        
        super(ImgModel, self).__init__()
        self.hidden_units = hidden_units
        self.arch_type = arch
        if (self.arch_type == 'vgg'):
            self.model = models.vgg19(pretrained=True)
            self.input_node=25088
        elif (self.arch_type == 'densenet'):
            self.model = models.densenet121(pretrained=True)
            self.input_node=1024

        
        for param in self.model.parameters():
            param.requires_grad = False
        

        self.model.classifier = nn.Sequential(
                            nn.Linear(self.input_node, self.hidden_units),
                            nn.ReLU(),
                            nn.Linear(self.hidden_units, 102),
                            nn.LogSoftmax(dim=1)
        )
        



    def forward(self, batch):        

        return self.model(batch) 

    @staticmethod
    def load(model_path: str):
        
        params = torch.load(model_path)
        model = params['model']
        model.load_state_dict(params['state_dict'])
        
        return model

    def save(self, path: str):
        """ Save the model to a file.
        @param path (str): path to the model
        """
        print('save model parameters to {}'.format(path))

        params = {
                'model': self.cpu(),
                'state_dict': self.state_dict()
        }
        
                      
        torch.save(params, path)
    