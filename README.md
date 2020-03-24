# RotNet

## 순서

1. cifar 10 dataset을 구성함.
2. efficientNet b0, b5 로 supervised learning
3. resnet50, 101으로 supervised learning
4. cifar 10 dataset을 변형하여 self-supervised learning 을 할 수 있도록 변형
- 한 데이터셋에 4개의 각도를 뒤집은 것이 같이 들어가도록 함
- 여러가지 argumentation을 줄 수 있도록 함
5. naver clova fasion dataset 도 수행
6. cutmix, langer optim 등 정확도 향상에 도움을 줄 수 있는 옵션들 추가
7. efficientNet backbone, resnet backbone 에서 self supervised learning 수행
8. 각 feature map 별로 classfier을 학습시킴
9. 2,3 번과 비교한 결과, 논문과 비교한 결과를 리포트

cheeta의 2020kaist volume 에 작업함. 

## 환경구성

환경은 cuda10.0, cudnn 7.5로 작업하였음

pytorch 는 cuda 10.0 에 맞는 1.4.0을 설치함

torchvision == 0.5.0

    pip install https://download.pytorch.org/whl/cu100/torch-1.4.0%2Bcu100-cp36-cp36m-linux_x86_64.whl

## Repo structor

    rotnet/
    │
    ├── train.py - main script to start training
    ├── test.py - evaluation of trained model
    |
    ├── requirements.txt - requirements
    │
    ├── parse_config.py - class to handle config file and cli options
    │
    ├── data_loader/ - anything about data loading goes here
    │   └── cifar10.py
    |   └── preprocess.py
    |   └── fassion.py
    |
    ├── configs/ - train config files
    |
    ├── model/ - models, losses, and metrics
    │   ├── model.py
    │   ├── metric.py
    │   └── loss.py
    |   └── optim.py
    │
    ├── saved/
    │   ├── models/ - trained models are saved here
    │   └── log/ - default logdir for tensorboard and logging output
    │
    ├── logger/ - module for tensorboard visualization and logging
    │  
    └── utils/ - small utility functions
        ├── util.py
        └── ...

## Dataset

dataset은 clova 의 fasion dataset과, cifar10 dataset을 이용함.

## Dataloader

[https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10](https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10)

[https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

위 두 링크를 참조하여 제작

unsupervised learning을 위해 nsuper=True 옵션을 키면 0, 90, 180, 270도 돌린사진과 각 라벨이 0,1,2,3 으로 구성된 데이터셋을 뽑을 수 있도록 제작함.

## Training

### config

config 파일을 준비하고, 그 파일을 통해 training을 할 수 있음

.ini 파일로 아래와 같이 적고 config 파일을 구성할 수 있음

    [EXP]
    seed = seed of your dataset(should be same for all exp)
    exp_name = #experiment name, should be unique
    batch_size = #size of batch
    model = {resnet, efficientnet}
    epochs = #number of epoch
    lr = #learning rate
    val_freq = #validation frequency(how often you validate your validation set if 1, eval every epoch
    worker = #number of workers 
    gpus = 0,1
    unsuper = {true}
    
    [DATASET]
    root = #root of dataset dir
    dataset = {cifar10, fasion}
    autoaug = #need to work 
    resize = #need to work
    hflip = #need to work
    rot = #need to work
    
    [MODEL]
    depth = {0,1,2,3,4,5,6,7 for efficientnet, 18, 50, 101, 152 for resnet}
    optim = {adam, rangerlars}
    scheduler = {reducelr. cosine, cyclic}
    criterion = {crossentropy}
    transfer = #need to work
    cutmix_alpha = {0 to 1}
    cutmix_prob = 0
    labelsmooth = #need to work

### [train.py](http://train.py) 실행

    python train.py --field {conifg field file direction} --config {config file direction}

을 통해 실행할 수 있으며, 

    tensorboard --logdir={logdir}

을 통해 텐서보드에서 train loss 와 vall acc, loss를 확인할 수 있음

해당 config file 의 exp name으로 log, moel 각각에 정보가 저장되며, model/{exp_name} 아래에 해당 train.py를 실행시킬 때 사용된 config file이 저장됨.

### Unsupervised learning

dataset 을 unsuper=True 옵션을 키면 0, 90, 180, 270도 돌린사진과 각 라벨이 0,1,2,3 으로 구성된 데이터셋을 뽑을 수 있음. config 에 unsuper = True를 키면 해당 데이터셋을 구성할 수 있음

### transfer learning(down stream task)

Unsupervised learning 을 통해 학습된 모델의 parameter을 고정하고 fc layer만 학습시키도록 하여 down stream task를 수행할 수 있음

    need to fill

### 정확도를 올리기 위해 사용한 기법들

- Label smoothing
- cutmix augmentation
- rangerlars optimizer
- learning rate scheduler : cyclic, cosine, reducelr

## Inference

train 시킨 모델에 몇개의 샘플을 test.py를 통해 확인할 수 있음.

## Result

## Issues

1. cifar 10은 32*32이미지인데, efficientnet 은 최소 224*224 이미지에 대해서 학습이 잘 되도록 설계되었음.
이걸 어캐 적절히 바꿔줘야할까?
[https://github.com/lukemelas/EfficientNet-PyTorch/issues/42](https://github.com/lukemelas/EfficientNet-PyTorch/issues/42)
2. unsupervised learning 방식을 평가하기 위한 적절한 metric과 dataset은 무엇인가?
3. configuration 기반 exp 관리를 깔끔하게 하는 방법은?

## To-do

- [x]  valset preprocessing 수정
- [x]  dataloader의 다양한 transform 추가(data augmentation)
- [x]  unsupervised learning 을 위한 dataset 준비
- [x]  fasion dataset(naver clova 제공) dataset 구현 - 기존 코드에 그대로 쓸 수 있도록
- [ ]  inference code
- [ ]  feature map을 뽑아서 바로 fc layer를 붙여 학습할 수 있도록 코드 구현
- [ ]  self-supervised learning 코드 작성 
- unsuper option 하나로 바로 training 까지

### Refference

- [https://github.com/wbaek/theconf](https://github.com/wbaek/theconf)
- [https://arxiv.org/abs/1905.04899](https://arxiv.org/abs/1905.04899)
- [https://github.com/mgrankin/over9000](https://github.com/mgrankin/over9000)