import re
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as m

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))


class m_AlexNet(nn.Module):
    def __init__(self):
        super(m_AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2 * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 2 * 2)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


def get_model(model_name):
    """
    :param model_name: alexnet vgg16 resnet50 resnet18
    :return: model instance
    """
    model = None

    if model_name == 'resnet50':
        model = m.resnet50()
    elif model_name == 'vgg16':
        model = m.vgg16()
    elif model_name == 'alexnet':
        model = m_AlexNet()
    elif model_name == 'resnet18':
        model = m.resnet18()
    else: 
        raise ValueError("No such model! \n")
    
    model.eval()
    return model
