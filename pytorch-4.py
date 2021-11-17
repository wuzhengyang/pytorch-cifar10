import torch
import torch.nn as nn
import numpy as np
import torch.functional as F
import torchvision.datasets
import torchvision.transforms as transforms
from collections import Counter

#定义超参数
BATCHSIZE=100
EPOCHES=20
LR=0.02

cfg = {
    'vgg16':[64,64,'m',128,128,'m',256,256,256,'m',512,512,512,'m',512,512,512,'m'],
    'vgg19':[64,64,'m',128,128,'m',256,256,256,256,'m',512,512,512,512,'m',512,512,512,512,'m']
}

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self.make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self,x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    def make_layers(self,cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'm':
                layers += [nn.MaxPool2d(kernel_size=2, stride = 2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding = 1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace = True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1,stride = 1)]
        return nn.Sequential(*layers)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='D:/OneDrive - zju.edu.cn/Work/code_deeplearning/data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='D:/OneDrive - zju.edu.cn/Work/code_deeplearning/data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
net = VGG('vgg16')
mlps = [net.to(device)]

optimizer = torch.optim.Adam([{"params": mlp.parameters()} for mlp in mlps], lr=LR)

loss_function = nn.CrossEntropyLoss()

for ep in range(EPOCHES):
    for img, label in trainloader:
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()  # 10个网络清除梯度
        for mlp in mlps:
            mlp.train()
            out = mlp(img)
            loss = loss_function(out, label)
            loss.backward()  # 网络们得到梯度
        optimizer.step()

    pre = []
    vote_correct = 0
    mlps_correct = [0 for i in range(len(mlps))]
    for img, label in testloader:
        img, label = img.to(device), label.to(device)
        for i, mlp in enumerate(mlps):
            mlp.eval()
            out = mlp(img)

            _, prediction = torch.max(out, 1)  # 按行取最大值
            pre_num = prediction.cpu().numpy()
            mlps_correct[i] += (pre_num == label.cpu().numpy()).sum()

            pre.append(pre_num)
        arr = np.array(pre)
        pre.clear()
        result = [Counter(arr[:, i]).most_common(1)[0][0] for i in range(BATCHSIZE)]
        vote_correct += (result == label.cpu().numpy()).sum()
    # print("epoch:" + str(ep)+"集成模型的正确率"+str(vote_correct/len(testloader)))

    for idx, coreect in enumerate(mlps_correct):
        print("Epoch： " + str(ep) + "  VGG正确率为：" + str(coreect / len(testloader)))