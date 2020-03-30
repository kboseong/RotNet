# RotNet

## 논문리뷰
[논문리뷰](https://github.com/kboseong/RotNet/blob/master/src/paper_review.md)

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

    trainset = SimpleImageLoader(root = {root dir}, split = 'unlabel', transform = {transform function}, unsuper = {True, False}, num_imgs_per_cat={#img per class})
    valset = SimpleImageLoader(root = {root dir}, split = 'validation', transform = {transform function}, unsuper= {True, False})
    num_classes = trainset.classnumber
    trainloader = torch.utils.data.DataLoader(trainset, batch_size={batch size}, shuffle=True, num_workers={#workers})
    valloader = torch.utils.data.DataLoader(valset, batch_size={batch size}, shuffle=False, num_workers={#workers})

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
    num_imgs_per_cat = {number of image for each class}
    type = #need to work
    autoaug = #need to work 
    resize = #need to work
    hflip = #need to work
    rot = #need to work
    
    [MODEL]
    depth = {0,1,2,3,4,5,6,7 for efficientnet, 18, 50, 101, 152 for resnet}
    optim = {adam, rangerlars}
    scheduler = {reducelr. cosine, cyclic}
    criterion = {crossentropy}
    transfer = {model weight location to transfer learning}
    block_op = {0~15 int}
    no_head = {true}
    cutmix_alpha = {0 to 1}
    cutmix_prob = 0
    labelsmooth = #need to work

### train 실행

    python train.py --field {conifg field file direction} --config {config file direction}

을 통해 실행할 수 있으며, 

    tensorboard --logdir=saved/logs

을 통해 텐서보드에서 train loss 와 vall acc, loss를 확인할 수 있음

해당 config file 의 exp name으로 saved 폴더 안에 log, moel 각각에 정보가 저장되며, model/{exp_name} 아래에 해당 train.py를 실행시킬 때 사용된 config file이 저장됨.

### Unsupervised learning

dataset 을 unsuper=True 옵션을 키면 0, 90, 180, 270도 돌린사진과 각 라벨이 0,1,2,3 으로 구성된 데이터셋을 뽑을 수 있음. config 에 unsuper = True를 키면 해당 데이터셋을 구성할 수 있음

### transfer learning(down stream task)

Unsupervised learning 을 통해 학습된 모델을 transfer, block_op 과 no_head 세가지 옵션을 통해 불러올 수 있음

transfer은 어떤 모델파일로 부터 weight를 로드할것인가에 대한 parameter임.

efficientnetb0는 block이 15개로, block_op을 0~15까지 줄 수 있으며, 이 옵션을 통해 해당 block까지만 가진 model을 생성할 수 있음.

original efficientnet의 경우 block 맨 끝에 head conv라 불리는 convolution layer하나가 추가되어있는데, 이 head는 최종 block의 output채널을 input channel로 받아 output channel=1280으로 내뱉음. 해당 layer를 추가할지 여부를 no_head option을 설정가능함

    from model.model import efficientnet
    #how to use
    model = efficientnet(phi = phi, num_classes = num_classes, transfer = transfer, block_num = block_num, no_head=True)

### 정확도를 올리기 위해 사용한 기법들

- Label smoothing
- cutmix augmentation
- rangerlars optimizer
- learning rate scheduler : cyclic, cosine, reducelr

위의 다양한 방법론들을 efficientnet b0 baseline의 정확도를 높이는데 사용하려 했으나, Label smoothing 과 cutmix augmentation 두가지는 이후 training 에서 조건을 같이 맞추기 위해 최종실험에서는 제외함.

## Inference

Need to work

## Experiment

### 환경

- Model = EfficientNet B0
- Dataset = Fasion Dataset (265 class fasion dataset from naver)
- Batch Size = 64
- Learning rate
    - supervised = 0.001
    - unsupervised = 0.01
- Learning rate scheduler = CosineAnnealingLR (시작 lr 부터 0까지 cosine함수로 줄어듬)
- Optimizer = RangerLars(RAdam + LARS + Lookahead) (현재 최고 성능 optimizer)
- DataAug
    - supervised train = RandomResizeCrop, RandomHorizontalFlip, RandomRotation, Normalize
    - unsupervised train = RandomResizeCrop, Normalize
    - All validation = Resize, Centercrop, Normalize

### 순서

1. Baseline을 위하여 Efficientnet 을 class 별 이미지 개수를 다르게 하여 학습시킴(5,10,20,30,40,50,60,70)
2. rotation task를 학습시킴 - unsupervised learning model
3. block을 2, 4, 10, 15에서 추출하고 각각에 대해서 conv layer + fc layer, fc layer 두가지 종류의 classifier을 붙여서 classification을 수행시킴 
4. rotation task 의 정확도 별로 weight를 다르게 가져와서 block, class별 이미지 개수, header를 고정시킨 상태에서 classification을 수행시킴
5. pretrained model weight를 고정한 상태에서 class별 이미지 개수를 달리면서 classfication을 수행시킴

아래의 실험들을 진행함

    #base line
    fasion_efficientnet_cat_5
    fasion_efficientnet_cat_10 
    fasion_efficientnet_cat_20 
    fasion_efficientnet_cat_30
    fasion_efficientnet_cat_40  
    fasion_efficientnet_cat_50
    fasion_efficientnet_cat_60
    fasion_efficientnet_cat_70
    
    fasion_efficientnet_cat_5_batch_64
    ****fasion_efficientnet_cat_10_batch_64
    fasion_efficientnet_cat_20_batch_64
    fasion_efficientnet_cat_30_batch_64
    fasion_efficientnet_cat_40_batch_64
    fasion_efficientnet_cat_50_batch_64
    fasion_efficientnet_cat_60_batch_64
    fasion_efficientnet_cat_70_batch_64
    
    #unsupervised learning
    fasion_efficientnet_b0_unsuper_no_full_batch(lr = 1e-2)
    fasion_efficientnet_b0_unsuper_no_full_batch_lr_1e-3
    
    #block test with head true
    POST_fasion_block_2_cat_70_head_true
    POST_fasion_block_4_cat_70_head_true
    POST_fasion_block_10_cat_70_head_true
    POST_fasion_block_15_cat_70_head_true(same stucture with efficientnet)
    POST_fasion_full_transfer_cat_70(same stucture with efficientnet, only different fc layer weight)
    
    #block test with head false
    POST_fasion_block_2_cat_70_head_false
    POST_fasion_block_4_cat_70_head_false
    POST_fasion_block_10_cat_70_head_false
    POST_fasion_block_15_cat_70_head_false
    
    #cat image test
    POST_fasion_block_15_cat_5_head_true
    POST_fasion_block_15_cat_10_head_true
    POST_fasion_block_15_cat_20_head_true
    POST_fasion_block_15_cat_30_head_true
    POST_fasion_block_15_cat_40_head_true
    POST_fasion_block_15_cat_50_head_true
    POST_fasion_block_15_cat_60_head_true
    POST_fasion_block_15_cat_70_head_true
    
    POST_fasion_full_transfer_cat_5
    POST_fasion_full_transfer_cat_10
    POST_fasion_full_transfer_cat_20
    POST_fasion_full_transfer_cat_30
    POST_fasion_full_transfer_cat_40
    POST_fasion_full_transfer_cat_50
    POST_fasion_full_transfer_cat_60
    POST_fasion_full_transfer_cat_70
    
    #rotnet-acc based exp
    POST_fasion_rotnet_0_block_15_cat_10_head_true
    POST_fasion_rotnet_3_block_15_cat_10_head_true
    POST_fasion_rotnet_4_block_15_cat_10_head_true
    POST_fasion_rotnet_7_block_15_cat_10_head_true
    POST_fasion_rotnet_11_block_15_cat_10_head_true
    POST_fasion_rotnet_17_block_15_cat_10_head_true
    POST_fasion_rotnet_36_block_15_cat_10_head_true
    POST_fasion_rotnet_63_block_15_cat_10_head_true
    POST_fasion_rotnet_98_block_15_cat_10_head_true
    
    #retraining from rotnet 4
    POST_fasion_rotnet_4_block_15_cat_5_head_true
    POST_fasion_rotnet_4_block_15_cat_10_head_true
    POST_fasion_rotnet_4_block_15_cat_20_head_true
    POST_fasion_rotnet_4_block_15_cat_30_head_true
    POST_fasion_rotnet_4_block_15_cat_40_head_true
    POST_fasion_rotnet_4_block_15_cat_50_head_true
    POST_fasion_rotnet_4_block_15_cat_60_head_true
    POST_fasion_rotnet_4_block_15_cat_70_head_true

## Result

1. Baseline을 위하여 Efficientnet 을 class 별 이미지 개수를 다르게 하여 학습시킴(5,10,20,30,40,50,60,70)

    ![](https://github.com/kboseong/RotNet/blob/master/src/efficientnet.png)

2. rotation task를 학습시킴 - unsupervised learning model

    → 94.8%의 최종 accuracy를 확보함

    ![](https://github.com/kboseong/RotNet/blob/master/src/acc_of_rotation_task_epoch.png)

3. block을 2, 4, 10, 15에서 추출하고 각각에 대해서 conv layer + fc layer, fc layer 두가지 종류의 classifier을 붙여서 classification을 수행시킴 

    → conv layer + fc lyaer가 단독 fc layer보다 성능이 높았고, block이 뒤쪽으로 갈수록 더 성능이 높았음
    → 이를 통해 head = true, block = 15 option을 이후에 사용하기로 함

    ![](https://github.com/kboseong/RotNet/blob/master/src/blocks_and_head.png)

4. rotation task 의 정확도 별로 weight를 다르게 가져와서 block, class별 이미지 개수, header를 고정시킨 상태에서 classification을 수행시킴

    → 58%정확도인 epoch 5 모델을 가져다 썼을 때 가장 성능이 높았으며, 그 이후로 성능이 떨어짐
    → epoch 5에서의 모델을 이후에 쓰기로 결정

    ![](https://github.com/kboseong/RotNet/blob/master/src/acc_of_supervised_by_unsuper_epoch.png)
    ![](https://github.com/kboseong/RotNet/blob/master/src/semisupervised_for_10_image.png)


5. pretrained model weight를 고정한 상태에서 class별 이미지 개수를 달리면서 classfication을 수행시킴

    → 기존 efficientnet b0 baseline과 비교하여 class 별 이미지 개수가 낮을 수록 supervised learning과 semi supervised learning의 차이가 컸음.

    ![](https://github.com/kboseong/RotNet/blob/master/src/semisup_and_sup_by_image_per_class.png)


## Discussion

### Question-1

alexnet을 이용한 기존 논문에는 앞쪽 block에서 image의 semantic한 feature을 뽑아내어 앞쪽 block을 사용하는 것이 더 좋은 성능을 보였으나 efficientnet의 경우는 그렇지 않음.

### Answer-1

Efficientnet은 논문에서 사용한 네트워크에 비해 깊은 구조를 가지며, 특정 image size의 input에 대해 classification을 가장 잘 수행할 수 있는 네트워크의 형태를 autoML 방식을 통해 찾은 모델임. 따라서, 앞쪽의 block까지만 사용할 수록, 그만큼 네트워크의 해석력이 원래의 네트워크에 비해 떨어지는 문제로 이어진 것으로 파악됨. 즉, 특정 block에서 이미지의 semantic한 feature을 잘학습했다고 하더라도, 원래의 task(265 classification)을 수행하기 위해서는 이후의 layer들의 역할이 중요하기 때문에, 네트워크의 형태를 가장 보존하는 block 15를 사용하는 것이 가장 효과적인 것으로 생각함.

### Question-2

왜 2, 4, 10 ,15 block을 선택했는가?

### Answer-2

feture map의 사이즈별로 마지막 block을 선택함. feature map이 (56,56), (28,28), (14,14), (7,7)로 줄어드는데, 줄어드는 마지막 block을 선택하였음.

### Question-3

왜 rotation task의 정확도가 높은 것보다 50~60% 사이의 구간이 가장 이후 fasion classification task를 더 잘 수행하는가?

### Answer-3

EfficientNet의 15block까지의 weight를 로드 하였을 때 참조되는 모델의 깊이가 굉장히 깊음. 따라서 rotnetation task에 너무 overfitting이 되면 오히려 본 task에 더 낮은 정확도를 보이는 것으로 생각됨


## Issues

1. cifar 10은 32*32이미지인데, efficientnet 은 최소 224*224 이미지에 대해서 학습이 잘 되도록 설계되었음.
이걸 어떻게 적절히 바꿔줘야할까?
[https://github.com/lukemelas/EfficientNet-PyTorch/issues/42](https://github.com/lukemelas/EfficientNet-PyTorch/issues/42)
2. unsupervised learning 방식을 평가하기 위한 적절한 metric과 dataset은 무엇인가?
3. unsupervised learning과 classification task 를 위한 각각의 train dataset augmentation 기법은 어떤 것들이 적절할까? 
4. configuration 기반 exp 관리를 깔끔하게 하는 방법은?
5. efficientnet 마지막의 con layer의 역할은?
6. 실험에 영향을 줄수있는 lr 등의 요소들이 많은데, 시간의 한계로 전부 하지 못하는 상황, 적절한 lr 선정은 어떻게?

## To-do

- [x] valset preprocessing 수정
- [x] dataloader의 다양한 transform 추가(data augmentation)
- [x] unsupervised learning 을 위한 dataset 준비
- [x] fasion dataset(naver clova 제공) dataset 구현 - 기존 코드에 그대로 쓸 수 있도록
- [x] feature map을 뽑아서 바로 fc layer를 붙여 학습할 수 있도록 코드 구현
- [x] self-supervised learning 코드 작성 
- [x] unsuper option 하나로 바로 training 까지
- [ ] inference code

### Refference

- [https://github.com/wbaek/theconf](https://github.com/wbaek/theconf)
- [https://arxiv.org/abs/1905.04899](https://arxiv.org/abs/1905.04899)
- [https://github.com/mgrankin/over9000](https://github.com/mgrankin/over9000)
- [https://github.com/victoresque/pytorch-template](https://github.com/victoresque/pytorch-template)
- [https://github.com/lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
- [https://github.com/gidariss/FeatureLearningRotNet](https://github.com/gidariss/FeatureLearningRotNet)
- [https://github.com/shine0624/semi-sup-learning](https://github.com/shine0624/semi-sup-learning)

