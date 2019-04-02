from __future__ import division
from torchvision import models
from torchvision import transforms
from PIL import Image
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np

CONTENT_PATH = "./images/Dorothea.jpeg"
STYLE_PATH = "./images/style1.jpg"
LEARNING_RATE = 0.01
TOTAL_STEP = 1000
STYLE_WEIGHT = 0.5
SPACE = 100



class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        #获取以下五个层的特征
        self.select = ['0', '5', '10', '19', '28']
        self.vgg = models.vgg19(pretrained=True).features

    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            #非常重要，必须是x = layer(x)
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features


class Main():
    #加载图片函数
    def loadImage(self,image_path, transform=None, max_size=None, shape=None):
        """加载图像，并进行Resize、transform操作"""
        image = Image.open(image_path)

        if max_size:
            scale = max_size / max(image.size)
            size = np.array(image.size) * scale
            image = image.resize(size.astype(int), Image.ANTIALIAS)

        if shape:
            image = image.resize(shape, Image.LANCZOS)

        if transform:
            image = transform(image).unsqueeze(0)

        return image.cuda()

    def mainFunc(self):

        #将图片转换为tensor且标准化
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        # print(input)
        self.content = self.loadImage(CONTENT_PATH,transform,max_size = 600)
        #content的长和宽
        shape = [self.content.size(2),self.content.size(3)]
        #使得content与style形状一致
        self.style = self.loadImage(STYLE_PATH,transform,shape = shape)

        target = self.content.clone().requires_grad_(True)
        # target = self.content.clone()
        optimizer = torch.optim.Adam([target], lr = LEARNING_RATE, betas=[0.5, 0.999])

        vgg = VGGNet().cuda().eval()
        # print(target)
        for step in range(TOTAL_STEP):
            target_features = vgg(target)
            content_features = vgg(self.content)
            style_features = vgg(self.style)
            content_loss = 0.0
            style_loss = 0.0
            for f1, f2, f3 in zip(target_features, content_features,
                                  style_features):
            # 计算content_loss
                content_loss += torch.mean((f1 - f2)**2)
                n, c, h, w = f1.size() #将特征reshape成二维矩阵相乘，求gram矩阵
                f1 = f1.view(c, h * w)
                f3 = f3.view(c, h * w)
                f1 = torch.mm(f1, f1.t())
                f3 = torch.mm(f3, f3.t())
                #计算style_loss
                style_loss += torch.mean((f1 - f3)**2) / (c * h * w) #计算总的loss
            loss = content_loss + style_loss * STYLE_WEIGHT
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step + 1) % SPACE == 0:
                print("test")
                denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
                img = target.clone().cpu().squeeze()
                img = denorm(img.data).clamp_(0, 1)
                torchvision.utils.save_image(img, './results/output-%d.png' % (step + 1))

if __name__ == "__main__":
    t = Main()
    t.mainFunc()