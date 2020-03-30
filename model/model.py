import torch
from efficientnet_pytorch import EfficientNet
from .classifier import classifier, classifier_with_no_head_conv

cut = ['_blocks.3._expand_conv.weight', '_blocks.5._expand_conv.weight', '_blocks.11._expand_conv.weight','_conv_head.weight' ]

def efficientnet(phi, num_classes, transfer=False, block_num = False, freeze = False, no_head = False):
    if phi in [0,1,2,3,4,5,6,7]:
        model = EfficientNet.from_name('efficientnet-b{}'.format(phi), override_params={'num_classes': num_classes}) 
    else:
        raise ValueError('EfficientNet should select phi in [0,1,2,3,4,5,6,7]')
    if transfer :
        print('load model weight from {}'.format(transfer))
        del model
        if no_head:
            print('you select no-head option')
            model = classifier_with_no_head_conv.from_name('efficientnet-b{}'.format(phi), override_params={'num_classes': num_classes}, block_num = int(block_num))
        else:
            model = classifier.from_name('efficientnet-b{}'.format(phi), override_params={'num_classes': num_classes}, block_num = int(block_num))
        pretrained_dict = torch.load(transfer, map_location='cpu')
        new_model_dict = model.state_dict()
        if (int(block_num)>=0) and (int(block_num) <= 15):
            new_dict = {}
            for k, v in pretrained_dict.items():
                
                if int(block_num) >= 15:
                    if k=='_conv_head.weight':
                        break
                    # have to feel
                if k.split('.')[1] == str(int(block_num)+1):
                    #print(k)
                    break
                    

                new_dict[k] = v
        else:
            new_dict = pretrained_dict  
            new_dict['_fc.weight'] = new_model_dict['_fc.weight']
            new_dict['_fc.bias'] = new_model_dict['_fc.bias']
            #raise ValueError('block op sould be in [0 to 15]')
        new_model_dict.update(new_dict)
        model.load_state_dict(new_model_dict)
    return model

def resnet(depth, num_classes):

    if depth in [18,34,50,101,152]:
        model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet{}'.format(depth), num_classes = num_classes, pretrained=False)
    else :
        raise ValueError('ResNet should select depth in [18, 34, 50, 101, 152]')
    return model