import foolbox
import numpy as np
import torchvision.models as models
import torch.nn as nn
import torch
import resnet.resnet
import os
def GetA():
    model = models.resnet18(pretrained=True).eval()
    model.fc=nn.Linear(model.fc.in_features,10)
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=10, preprocessing=preprocessing)

    images, labels = foolbox.utils.samples(dataset='cifar10', batchsize=16, data_format='channels_first', bounds=(0, 1))
    # print(np.mean(fmodel.forward(images).argmax(axis=-1) == labels))

    attack = foolbox.attacks.FGSM(fmodel)
    adversarials = attack(images, labels)
    print(len(adversarials[0][0][0]))
    print(len(adversarials[0][0]))
    print(len(adversarials[0]))
    print(len(adversarials))


    return adversarials,labels

def GetAcc(model, images, labels, device):
    correct_val = 0.
    total_val = 0.
    inputs = images
    ip = torch.tensor(inputs)
    lb = torch.tensor(labels)
    ip, lb = ip.to(device), lb.to(device)

    outputs = model(ip)
    _, predicted = torch.max(outputs.data, 1)
    total_val += len(labels)
    correct_val += (predicted == lb).squeeze().cpu().sum().numpy()
    print(correct_val/total_val)

# a,b = GetA()
# count = 0
# for i1 in a:
#     for i2 in i1:
#         for i3 in i2:
#             np.savetxt('AttackData/'+ str(count) + ".txt", i3, fmt="%float32", delimiter=',') #保存为7位小数的浮点数，用逗号分隔
#             count+=1
# a3 = [0]*32
# a2 = [a3]*32
# a1 = [a2]*3
# AttackDataArray = a1*16
# tempC = 0
# for root, dirs, files in os.walk("AttackData/"):
#     for index1 in range(0,16):
#         for index2 in range(0,3):
#             for index3 in range(0,32):
#                 for index4 in range(0,32):
#                     ct = index4+index3
#                     AttackDataArray[index1][index2][index3][index4] = files[]
#
#
# for root, dirs, files in os.walk("AttackData/"):
#     for filename in files:
#         with open(filename) as f:
#             for line in f:
#                 if tempCount%len(a[0][0][0]) == 0:
#                     a3.extend(line)
#
#                     print(line, end='')
#                     tempCount += 1
#
#         # GetA()