import pickle as p
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plimg
from PIL import Image


def load_CIFAR_batch(filename):
    with open(filename, 'rb')as f:
        #       datadict = p.load(f)
        datadict = p.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32)
        Y = np.array(Y)
        return X, Y


def load_CIFAR_Labels(filename):
    with open(filename, 'rb') as f:
        lines = [x for x in f.readlines()]
        print(lines)


if __name__ == "__main__":
    load_CIFAR_Labels("./data/cifar-10-batches-py/batches.meta")
    imgX, imgY = load_CIFAR_batch("./data/cifar-10-batches-py/data_batch_1")
    print(imgX.shape)
    print("正在保存图片:")
    #   for i in range(imgX.shape[0]):
    for i in range(128):  # 值输出10张图片，用来做演示
        #       imgs = imgX[i - 1]#?
        imgs = imgX[i]
        img0 = imgs[0]
        img1 = imgs[1]
        img2 = imgs[2]
        i0 = Image.fromarray(img0)  # 从数据，生成image对象
        i1 = Image.fromarray(img1)
        i2 = Image.fromarray(img2)
        img = Image.merge("RGB", (i0, i1, i2))
        name = "cifar10_" + str(i) +"_" + str(imgY[i]) + ".png"
        img.save("./cifar10_images/" + name, "png")  # 文件夹下是RGB融合后的图像

        # for j in range(imgs.shape[0]):
        #     #               img = imgs[j - 1]
        #     img = imgs[j]
        #     name = "img_" + str(i) +"_"+ str(j) + ".png"
        #     print("正在保存图片" + name)
        #     plimg.imsave("./cifar10_images/" + name, img)  # 文件夹下是RGB分离的图像

    print("保存完毕.")
