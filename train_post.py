import argparse
import numpy as np
import torch
import torch.optim as optim
from model.model import efficientnet, resnet
from dataloader.fasion import SimpleImageLoader
from dataloader.cifar10 import CIFAR10
from dataloader.preprocess import trainpreprocess, valpreprocess,rotpreprocess
from torchvision import transforms
import configparser
from tensorboardX import SummaryWriter
from parse_config import parse_config
from model.over9000.over9000 import RangerLars
from utils.util import rand_bbox
from model.metric import accuracy
from model.cyclicLR import CyclicLR
import os

def main(args):
    config = parse_config(args.field, args.config)
    
    # exp params
    seed = int(config['EXP']['seed'])
    exp_name = config['EXP']['exp_name']
    batch_size = int(config['EXP']['batch_size'])
    model_name = config['EXP']['model']
    epochs = int(config['EXP']['epochs'])
    lr = float(config['EXP']['lr'])
    val_freq = int(config['EXP']['val_freq'])
    worker = int(config['EXP']['worker'])
    gpus = config['EXP']['gpus']
    unsuper = config['EXP']['unsuper']


    # dataset params
    dataset = config['DATASET']['dataset']
    root = config['DATASET']['root']
    num_imgs_per_cat = config['DATASET']['num_imgs_per_cat']
    d_type = config['DATASET']['type']

    # gpu
    os.environ["CUDA_VISIBLE_DEVICES"]=gpus


    # model params
    optim_name = config['MODEL']['optim']
    scheduler_name = config['MODEL']['scheduler']
    criterion_name = config['MODEL']['criterion']
    transfer = config['MODEL']['transfer']
    cutmix_alpha = float(config['MODEL']['cutmix_alpha'])
    cutmix_prob = float(config['MODEL']['cutmix_prob'])
    labelsmooth = config['MODEL']['labelsmooth']

    # fix seed
    np.random.seed(seed)
    torch.manual_seed(seed) 

    if unsuper == 'true':
        unsuper = True
    else:
        unsuper = False
    # make dataloader
    if unsuper:
        rot_preprocess = rotpreprocess()
    train_preprocess = trainpreprocess(config['DATASET'])
    val_preprocess = valpreprocess()
    
    if dataset == 'fasion':
        trainset = SimpleImageLoader(root = root, split = 'train', transform = train_preprocess, num_imgs_per_cat=num_imgs_per_cat)
        valset = SimpleImageLoader(root = root, split = 'validation', transform = val_preprocess)
        num_classes = trainset.classnumber

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=worker)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=worker)

    # get model
    if model_name == 'efficientnet':
        phi = int(config['MODEL']['depth'])
        model = efficientnet(phi = phi, num_classes = num_classes, transfer = transfer, block_op = 0)
    else:
        raise ValueError('no supported model name')
    model = model.cuda()
    model = torch.nn.DataParallel(model).cuda()
    model.training = True

    # set loss & optimizer & scheduler
    if criterion_name == 'crossentropy':
        criterion = torch.nn.CrossEntropyLoss()
    else :
        raise ValueError('no supported loss function name')

    if optim_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optim_name == 'rangerlars':
        optimizer = RangerLars(model.parameters(), lr=lr)
    else :
        raise ValueError('no supported optimizer name')
    
    if scheduler_name =='reducelr':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    elif scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.)
    elif scheduler_name == 'cyclic':
        scheduler = CyclicLR(optimizer, base_lr=lr * 0.3, max_lr=lr, step_size_up=10, cycle_momentum=False)
    else :
        raise ValueError('no supported scheduler name')

    # save dir
    try: 
        os.mkdir('saved/models/{}'.format(exp_name))
        os.mkdir('saved/logs/{}'.format(exp_name))
    except:
        raise ValueError('existed exp name : {}'.format(exp_name))
    
    writer = SummaryWriter("saved/logs/{}".format(exp_name))

    #save config 
    with open('saved/models/{}/config.ini'.format(exp_name), 'w') as configfile:
        config_saved = configparser.ConfigParser(allow_no_value=True)
        config_saved.read(args.config)
        config_saved.write(configfile)

    # training 
    iter = 0
    best_acc = 0
    
    for epoch_num in range(epochs):

        # -------------------------------- train model ---------------------------- #
        model.train()
        epoch_loss = []
        for iter_num, data in enumerate(trainloader):
            optimizer.zero_grad()
            
            image, label = data[0], data[1]
            image = image.cuda()
            label = label.cuda()

            feature_maps = model.extract_features_midconv(image)
            loss = criterion(pred, label)

            if bool(loss == 0):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            epoch_loss.append(float(loss))

            writer.add_scalar('Loss/train', loss, iter)
            iter+=1
            print('Epoch: {} | Iteration: {} | Running loss: {:1.5f}'.format(epoch_num, iter_num, np.mean(np.array(epoch_loss))))
        
        # -------------------------------- validate model ---------------------------- #
        model.eval()
        val_loss = []
        top1_list = []
        top5_list = []
        with torch.no_grad():
            for iter_num, data in enumerate(valloader):
                image, label = data[0], data[1]
                image = image.cuda()
                label = label.cuda()

                pred = model(image)
                loss = criterion(pred, label)
                top1, top5 = accuracy(pred, label, (1, 5))
                
                top1_list.append(top1.item())
                top5_list.append(top5.item())
                #val_correct += (torch.max(output, 1)[1] == label).sum().item()
                val_loss.append(float(loss))

        #acc = val_correct / len(valset)
        val_loss_mean = np.mean(np.array(val_loss))
        top1 = np.mean(np.array(top1_list))
        top5 = np.mean(np.array(top5_list))
        
        if scheduler_name =='reducelr':
            scheduler.step(val_loss_mean)
        else:
            scheduler.step()

        print('Epoch: {} | loss : {:1.5f} | top1-acc : {:1.5f} | top5-acc : {:1.5f}'.format(epoch_num, val_loss_mean, top1, top5))

        if best_acc < top1 :
            best_acc = top1

        writer.add_scalar('Acc/top1-acc', top1, epoch_num)
        writer.add_scalar('Acc/top5-acc', top5, epoch_num)
        writer.add_scalar('Loss/val', val_loss_mean, epoch_num)
        writer.add_scalar('satus/lr', optimizer.param_groups[0]['lr'] , epoch_num)
        torch.save(model.module.state_dict(), 'saved/models/{}/model_{}.pt'.format(exp_name, epoch_num))
        
        torch.save(model.module.state_dict(), 'saved/models/{}/model_best_acc_{}.pt'.format(exp_name, best_acc))




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Simple training script for training a rotnet network.')

    #config
    parser.add_argument('--field', help='config field directory')  
    parser.add_argument('--config', help='config file directory')    
    args = parser.parse_args()

    main(args)
