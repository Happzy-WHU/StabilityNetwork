import torch
import torch.nn as nn
import os
from resnet import resnet,resnet_layer,resnet_size_kernel,resnet_drop,resnet_bias_true
BASEDIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置模型训练的设备

print("use device :{}".format(device))

net1, net2, net3, net4, net5, net6, net7, net8, net9 \
    = (resnet.ResNet18(), resnet.ResNet18(), resnet_bias_true.ResNet18(), resnet_drop.ResNet18(), resnet.ResNet18(),
       resnet.ResNet18(), resnet_layer.ResNet18(), resnet.ResNet18(), resnet_size_kernel.ResNet18())






classes = 10
num_ftrs = net1.fc.in_features  # 从原始的resnet18从获取输入的结点数
seq = nn.Linear(num_ftrs, classes)

def case1(state_dict_load):
    net1.load_state_dict(state_dict_load , False)
    net1.fc = seq
    net1.to(device)
    return net1

def case2(state_dict_load):
    net2.load_state_dict(state_dict_load , False)
    net2.fc = seq
    net2.to(device)
    return net2

def case3(state_dict_load):
    net3.load_state_dict(state_dict_load , False)
    net3.fc = seq
    net3.to(device)
    return net3

def case4(state_dict_load):
    net4.load_state_dict(state_dict_load , False)
    net4.fc = seq
    net4.to(device)
    return net4

def case5(state_dict_load):
    net5.load_state_dict(state_dict_load , False)
    net5.fc = seq
    net5.to(device)
    return net5

def case6(state_dict_load):
    net6.load_state_dict(state_dict_load , False)
    net6.fc = seq
    net6.to(device)
    return net6

def case7(state_dict_load):
    net7.load_state_dict(state_dict_load , False)
    net7.fc = seq
    net7.to(device)
    return net7

def case8(state_dict_load):
    net8.load_state_dict(state_dict_load , False)
    net8.fc = seq
    net8.to(device)
    return net8

def case9(state_dict_load):
    net9.load_state_dict(state_dict_load , False)
    net9.fc = seq
    net9.to(device)
    return net9


switch={
"adam.pth": case1,
"batch.pth": case2,
"bias.pth": case3,
"drop.pth": case4,
"early.pth": case5,
"first.pth": case6,
"layer.pth": case7,
"lr.pth": case8,
"size_kernel.pth": case9
}

def build_model():
    net_list = []
    for root, dirs, files in os.walk("./models/cifar10_models/"):
        for name in files:
            path_pretrained_model = "./models/cifar10_models/" + name
            state_dict_load = torch.load(path_pretrained_model)
            net_list.append(switch[name](state_dict_load))
    return net_list

# def newNet():
#     models = nn.Sequential()
#     models.add_module(nn.Conv2d(3, 64,kernel_size=7, stride=2, padding='same',bias=True))
#     models.add_module(nn.ReLU(inplace=True))
#     models.add_module(nn.BatchNorm2d)
#     models.add_module(nn.Dropout(0.5))
#
#     models.add_module(nn.Conv2d(128, (3, 3), padding='same'))
#     models.add_module(nn.ReLU(inplace=True))
#     models.add_module(nn.BatchNorm2d)
#     models.add_module(nn.Dropout(0.5))
#
#     models.add_module(nn.Conv2d(256, (3, 3), padding='same'))
#     models.add_module(nn.ReLU(inplace=True))
#     models.add_module(nn.BatchNorm2d)
#     x=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#     models.add_module(x)
#     models.add_module(nn.Linear())


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(  # (1,28,28)
                in_channels=10,
                out_channels=10,
                kernel_size=5,
                stride=1,
                padding=2  # padding=(kernelsize-stride)/2
            ),  # (16,28,28)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5)  # (16,14,14)
        )

        self.out = nn.Linear(10, 10) #(32,64)

    # 定义前向传播过程，过程名字不可更改，因为这是重写父类的方法
    def forward(self, x):
        x = self.conv1(x)
        x = x.contiguous().view(x.size(0), -1)  # (batch,32*7*7)
        output = self.out(x)
        return output


# build_model()