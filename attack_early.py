# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision
import resnet.resnet as resnet
import foolbox
import numpy as np
import attack_first

BASEDIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置模型训练的设备

print("use device :{}".format(device))


# 参数设置
MAX_EPOCH = 3
BATCH_SIZE = 64 #64
LR = 0.01
log_interval = 10
val_interval = 1
classes = 10 #类的数量，不可改
start_epoch = 0 #本质上和改Max_EPOCH等价
lr_decay_rate = 0.1
lr_decay_step = 60   #每60次迭代将学习率乘以0.1
describe = "new"

#<editor-fold desc="step 1/5 数据">
norm_mean = [0.4914, 0.4822, 0.4465]
norm_std = [0.2023, 0.1994, 0.2010]
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

trainset = torchvision.datasets.CIFAR10\
    (root='./data', train=True, download=True, transform=transform_train) #训练数据集
train_loader = torch.utils.data.DataLoader\
    (trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取

testset = torchvision.datasets.CIFAR10\
    (root='./data', train=False, download=True, transform=transform_test)
valid_loader = torch.utils.data.DataLoader\
    (testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


resnet18_ft = resnet.ResNet18()
# 2/3 加载参数
# flag = 0
flag = 1
if flag:
    path_pretrained_model = os.path.join(BASEDIR, "models/cifar10_models/early.pth")
    state_dict_load = torch.load(path_pretrained_model)
    resnet18_ft.load_state_dict(state_dict_load)

num_ftrs = resnet18_ft.fc.in_features
resnet18_ft.fc = nn.Linear(num_ftrs, classes)

# Cifar-10的标签
classes_detail = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#</editor-fold>

#<editor-fold desc="step 3/5 损失函数">

criterion = nn.CrossEntropyLoss()                                                   # 选择损失函数

#</editor-fold>

#<editor-fold desc="step 4/5 优化器">

def getScheduler(model):
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)  # 选择优化器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_rate)  # 设置学习率下降策略
    return optimizer,scheduler

#</editor-fold>

#<editor-fold desc="step 5/5 训练">
def getOutput(my_model,inputs):
    outputs = my_model(inputs)
    return outputs

resnet18_ft = resnet18_ft.cuda()

optimizer, scheduler = getScheduler(resnet18_ft)
def train_new():
    train_curve = list()
    valid_curve = list()
    for epoch in range(start_epoch + 1, MAX_EPOCH):
        loss_mean = 0.
        correct = 0.
        total = 0.
        resnet18_ft.train()
        for i, data in enumerate(train_loader):

            # forward
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # 训练数据也要放到设置的设备上
            midDataList = []
            outputs = resnet18_ft(inputs)

            # backward
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()

            # update weights
            optimizer.step()

            # 统计分类情况
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).squeeze().cpu().sum().numpy()

            # 打印训练信息
            loss_mean += loss.item()
            train_curve.append(loss.item())
            if (i + 1) % log_interval == 0:
                loss_mean = loss_mean / log_interval
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH, i + 1, len(train_loader), loss_mean, correct / total))
                loss_mean = 0.

                # if flag_m1:
                # print("epoch:{} conv1.weights[0, 0, ...] :\n {}".format(epoch, resnet18_ft.conv1.weight[0, 0, ...]))

        scheduler.step()  # 更新学习率

        # validate the model
        if (epoch + 1) % val_interval == 0:

            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            resnet18_ft.eval()
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = resnet18_ft(inputs)
                    loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).squeeze().cpu().sum().numpy()

                    loss_val += loss.item()

                loss_val_mean = loss_val / len(valid_loader)
                valid_curve.append(loss_val_mean)
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH, j + 1, len(valid_loader), loss_val_mean, correct_val / total_val))
            resnet18_ft.train()

    return resnet18_ft
if __name__ == "__main__":
    model = train_new()
    images , labels= attack_first.GetA()
    attack_first.GetAcc(model=model,images=images,labels=labels,device=device)



