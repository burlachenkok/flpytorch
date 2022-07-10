# ResNets from Torch Vision
from torchvision.models import resnet18 as tv_resnet18
from torchvision.models import resnet34 as tv_resnet34
from torchvision.models import resnet50 as tv_resnet50

# ResNets for CIFAR Datasets
from .resnet_cifar import resnet20
from .resnet_cifar import resnet32
from .resnet_cifar import resnet44
from .resnet_cifar import resnet56
from .resnet_cifar import resnet110
from .resnet_cifar import resnet1202

from .resnet_cifarlike import ResNet18 as resnet18_cifar
from .resnet_cifarlike import ResNet34 as resnet34_cifar
from .resnet_cifarlike import ResNet50 as resnet50_cifar
from .resnet_cifarlike import ResNet101 as resnet101_cifar
from .resnet_cifarlike import ResNet152 as resnet152_cifar

# VGGs for CIFAR Datasets
from .vgg_cifar import vgg11 as vgg11_cifar
from .vgg_cifar import vgg13 as vgg13_cifar
from .vgg_cifar import vgg16 as vgg16_cifar
from .vgg_cifar import vgg19 as vgg19_cifar

# WideResNets for CIFAR Datasets
from .wideresnet_cifar import WideResNet_28_2 as wideresnet282_cifar
from .wideresnet_cifar import WideResNet_28_4 as wideresnet284_cifar
from .wideresnet_cifar import WideResNet_28_8 as wideresnet288_cifar

# models for Shakespeare
from .rnn import rnn, minirnn

# models for FEMNIST
from .femnist import femnist, minifemnist

RNN_MODELS = ['rnn', 'minirnn']
