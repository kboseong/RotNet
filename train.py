import argparse
import numpy as np
import torch
import torch.optim as optim
import model
from dataloader import CIFAR10
from preprocessing import preprocess, valprocess
from torchvision import transforms
import configparser
from tensorboardX import SummaryWriter

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

    # make dataloader
    preprocess = preprocess()
    valprocess = valprocess()
    if dataset == 'cifar10':
        trainset = CIFAR10(root=root, train=True, download=True, transform=preprocess)
        valset = CIFAR10(root=root, train=False, download=True, transform=valprocess)
        num_classes = len(trainset.classes)
    else :
        raise ValueError('make sure dataset is cifar 10, etc')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=worker, seed=seed)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=worker, seed=seed)

    # get model
    if model_name == 'efficientnet':
        phi = config['MODEL']['phi']
        model = model.efficientnet(phi = phi, num_classes = num_classes)
    elif model_name == 'resnet':
        depth = config['MODEL']['depth']
        model = model.resnet(depth = depth, num_classes = num_classes)
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
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
    else :
        raise ValueError('no supported scheduler name')

    #tensorboard
    try: 
        os.mkdir('model/{}'.format(exp_name))
        os.mkdir('logs/{}'.format(exp_name))
    except:
        raise ValueError('existed exp name : {}'.format(exp_name))
    
    writer = SummaryWriter("logs/{}".format(exp_name))

    # training 
    iter = 0
    for epoch_num in range(epochs):
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

            epoch_loss.append(loss)

            writer.add_scalar('Loss/train', loss, iter)
            iter+=1
            print('Epoch: {} | Iteration: {} | Running loss: {:1.5f}'.format(epoch_num, iter_num, np.mean(np.array(epoch_loss))))



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Simple training script for training a rotnet network.')

    # dataset env
    parser.add_argument('--config', help='config file directory')    
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    main(config)