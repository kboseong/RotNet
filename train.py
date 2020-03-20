import argparse
import numpy as np
import torch
import torch.optim as optim
from  model import efficientnet, resnet
from dataloader import CIFAR10
from preprocessing import preprocess, valprocess
from torchvision import transforms
import configparser
from tensorboardX import SummaryWriter
import os

def main(config):
    dataset = config['EXP']['dataset']
    root = config['EXP']['root']
    seed = int(config['EXP']['seed'])
    exp_name = config['EXP']['exp_name']
    batch_size = int(config['EXP']['batch_size'])
    model_name = config['EXP']['model']
    epochs = int(config['EXP']['epochs'])
    lr = float(config['EXP']['lr'])
    val_freq = int(config['EXP']['val_freq'])
    worker = int(config['EXP']['worker'])
    gpus = config['EXP']['gpus']
    optim_name = config['MODEL']['optim']
    scheduler_name = config['MODEL']['scheduler']
    criterion_name = config['MODEL']['criterion']
    dataset_flag = config['EXP']['unsuper']

    # fix seed
    np.random.seed(seed)
    torch.manual_seed(seed) 

    # make dataloader
    pre_process = preprocess()
    val_process = valprocess()
    if dataset == 'cifar10':
        if dataset_flag == 'true':
            unsuper = True
        else:
            unsuper = False
        trainset = CIFAR10(root=root, train=True, download=True, transform=pre_process, unsuper = unsuper)
        valset = CIFAR10(root=root, train=False, download=True, transform=val_process, unsuper = unsuper)
        num_classes = len(trainset.classes)
    else :
        raise ValueError('make sure dataset is cifar 10, etc')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=worker)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=worker)

    # get model
    if model_name == 'efficientnet':
        phi = int(config['MODEL']['phi'])
        model = efficientnet(phi = phi, num_classes = num_classes)
    elif model_name == 'resnet':
        depth = int(config['MODEL']['depth'])
        model = resnet(depth = depth, num_classes = num_classes)
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
    else :
        raise ValueError('no supported optimizer name')
    
    if scheduler_name =='reducelr':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    else :
        raise ValueError('no supported scheduler name')

    # save dir
    try: 
        os.mkdir('model/{}'.format(exp_name))
        os.mkdir('logs/{}'.format(exp_name))
    except:
        raise ValueError('existed exp name : {}'.format(exp_name))
    
    writer = SummaryWriter("logs/{}".format(exp_name))

    #save config 
    with open('model/{}/config.ini'.format(exp_name), 'w') as configfile:
        config.write(configfile)

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

            output = model(image)

            loss = criterion(output, label)


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
        val_correct = 0
        with torch.no_grad():
            for iter_num, data in enumerate(valloader):
                image, label = data[0], data[1]
                image = image.cuda()
                label = label.cuda()

                output = model(image)
                loss = criterion(output, label)

                val_correct += (torch.max(output, 1)[1] == label).sum().item()
                val_loss.append(float(loss))

        acc = val_correct / len(valset)
        val_loss_mean = np.mean(np.array(val_loss))
        scheduler.step(val_loss_mean)

        if best_acc < acc :
            best_acc = acc

        writer.add_scalar('Acc/val', acc, epoch_num)
        writer.add_scalar('Loss/val', val_loss_mean, epoch_num)
        torch.save(model.module.state_dict(), 'model/{}/model_{}.pt'.format(exp_name, epoch_num))

    torch.save(model.module.state_dict(), 'model/{}/model_best_acc_{}.pt'.format(exp_name, best_acc))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Simple training script for training a rotnet network.')

    # dataset env
    parser.add_argument('--config', help='config file directory')    
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    main(config)