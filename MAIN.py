import torch
import torch.nn as nn
import torchvision
from torchvision import transforms,models
from PIL import Image
import argparse
import numpy as np
import os
import cv2 as cv

CONTENT_PATH = "./content.jpg"
STYLE_PATH = "./style.jpg"


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        #获取以下五个层的特征
        self.select = ['0', '5', '10', '19', '28']
        self.vgg19 = models.vgg19(pretrained=True).features

    def forward(self, x):
        features = []
        for name, layer in self.vgg19._models.items():
            feature = layer(x)
            if name in self.select:
                features.append(feature)

        return features


class Main():
    #加载图片函数
    def loadImage(self,path,transform,shape = None):
        image = cv.imread(path)
        if shape != None:
            image = cv.resize(src = image,dst = image,dsize = (shape[0],shape[1]))
        image_transformed = transform(image).unsqueeze(0)
        return image_transformed.cuda()

    def main(self):
        #将图片转换为tensor且标准化
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        # print(input)
        self.content = self.loadImage(CONTENT_PATH,transform)
        #content的长和宽
        shape = [self.content.size(3),self.content.size(2)]
        #使得content与style形状一致
        self.style = self.loadImage(STYLE_PATH,transform,shape)

if __name__ == "__main__":
    t = Main()
    t.main()