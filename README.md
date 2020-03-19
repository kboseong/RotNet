# RotNet

## 순서

1. cifar 10 dataset을 구성함.
2. efficientNet b0, b5 로 supervised learning
3. resnet50, 101으로 supervised learning
4. cifar 10 dataset을 변형하여 self-supervised learning 을 할 수 있도록 변형
- 한 데이터셋에 4개의 각도를 뒤집은 것이 같이 들어가도록 함
- 여러가지 argumentation을 줄 수 있도록 함
5. efficientNet backbone, resnet backbone 에서 self supervised learning 수행
6. 각 feature map 별로 classfier을 학습시킴
7. 2,3 번과 비교한 결과, 논문과 비교한 결과를 리포트

cheeta의 2020kaist volume 에 작업함. 

## 환경구성

환경은 cuda10.0, cudnn 7.5로 작업하였음

pytorch 는 cuda 10.0 에 맞는 1.4.0을 설치함

torchvision == 0.5.0

    pip install https://download.pytorch.org/whl/cu100/torch-1.4.0%2Bcu100-cp36-cp36m-linux_x86_64.whl

## dataset

[https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10](https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10)

[https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

위 두 링크를 참조하여 제작

## Training

### config

config 파일을 준비하고, 그 파일을 통해 training을 할 수 있음

.ini 파일로 아래와 같이 적고 config 파일을 구성할 수 있음

    [EXP]
    dataset = cifar10
    root = ../datasets
    seed = 777
    exp_name = efficientnet_b0
    batch_size = 64
    model = efficientnet
    epochs = 100
    lr = 1e-3
    val_freq = 1
    worker = 4
    gpus = 0
    
    [MODEL] 
    phi = 0
    optim = adam
    scheduler = reducelr
    criterion = crossentropy

### [train.py](http://train.py) 실행

    python train.py --config {config file direction}

을 통해 실행할 수 있으며, 

    tensorboard --logdir={logdir}

을 통해 텐서보드에서 train loss 와 vall acc, loss를 확인할 수 있음

## Inference

## Issues

1. cifar 10은 32*32이미지인데, efficientnet 은 최소 224*224 이미지에 대해서 학습이 잘 되도록 설계되었음.
이걸 어캐 적절히 바꿔줘야할까?
[https://github.com/lukemelas/EfficientNet-PyTorch/issues/42](https://github.com/lukemelas/EfficientNet-PyTorch/issues/42)

## To-do

- [ ]  valset preprocessing 수정
- [ ]  dataloader의 다양한 transform 추가(data augmentation)
- [ ]  inference code
- [ ]  unsupervised learning 을 위한 dataset 준비
- [ ]  feature map을 뽑아서 바로 fc layer를 붙여 학습할 수 있도록 코드 구현