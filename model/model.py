import torch
from efficientnet_pytorch import EfficientNet

cut = ['_blocks.3._expand_conv.weight', '_blocks.3._expand_conv.weight', '_blocks.3._expand_conv.weight','_conv_head.weight' ]

def efficientnet(phi, num_classes, transfer=False, block_op = False):
    if phi in [0,1,2,3,4,5,6,7]:
        model = EfficientNet.from_name('efficientnet-b{}'.format(phi), override_params={'num_classes': num_classes}) 
    else:
        raise ValueError('EfficientNet should select phi in [0,1,2,3,4,5,6,7]')
    if transfer :
        pretrained_dict = torch.load(transfer, map_location='cpu')

        if block_op in [0,1,2,3]:
            new_dict = {}
            for k, v in pretrained_dict.items():
                if k == cut[block_op]:
                    break
                new_dict[k] = v
        else:
            raise ValueError('block op sould be in [0,1,2,3]')

        model.load_state_dict(new_dict)
    return model

def resnet(depth, num_classes):

    if depth in [18,34,50,101,152]:
        model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet{}'.format(depth), num_classes = num_classes, pretrained=False)
    else :
        raise ValueError('ResNet should select depth in [18, 34, 50, 101, 152]')
    return model