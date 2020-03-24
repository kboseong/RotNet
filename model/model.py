import torch
from efficientnet_pytorch import EfficientNet


def efficientnet(phi, num_classes):
    if phi in [0,1,2,3,4,5,6,7]:
        model = EfficientNet.from_name('efficientnet-b{}'.format(phi), override_params={'num_classes': num_classes}) 
    else:
        raise ValueError('EfficientNet should select phi in [0,1,2,3,4,5,6,7]')
    return model

def resnet(depth, num_classes):

    if depth in [18,34,50,101,152]:
        model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet{}'.format(depth), num_classes = num_classes, pretrained=False)
    else :
        raise ValueError('ResNet should select depth in [18, 34, 50, 101, 152]')
    return model