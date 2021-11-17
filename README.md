# pytorch-cifar100
## 环境
python3.7
pytorch1.6.0+cu10.1
## 流程

**1、数据读取及预处理**

采用GPU对PyTorch进行速度提升，如果不存在GPU，则会自动选择CPU进行运算。

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

数据集的读取利用PyTorch的torchvision库，可以选择将download参数改为True进行下载。

数据集提前采用正则化的方式进行预处理，分为训练集和测试集，并采用生成器的方式加载数据，便于更好的处理大批量数据。classes为CIFAR-10数据集的10个标签类别。
```python
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
```
**2、构建VGGNet模型**

选择VGG16。VGGNet是通过对AlexNet的改进，进一步加深了卷积神经网络的深度，采用堆叠3 x 3的卷积层和2 x 2的降采样层，实现11到19层的网络深度。VGG的结构图如下所示。

![image-1](https://github.com/wuzhengyang/IMG/blob/main/image-1.png)

VGGNet模型总的来说，分为VGG16和VGG19两类，区别在于模型的层数不同，以下'M'参数代表池化层，数据代表各层滤波器的数量。

```python
cfg = {
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}
```

VGGNet模型的为全连接层，卷积层中都运用批量归一化的方法，提升模型的训练速度与收敛效率，并且可以一定的代替dropout的作用，有利于模型的泛化效果。
```python
class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
    
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
```
**3、定义模型超参数以及评估方法**

模型的学习率、训练次数、批次大小通过超参数的方式设定，优化函数采用Adam，损失函数采用交叉熵进行计算。

```python
LR = 0.001
EPOCHES = 20
BATCHSIZE = 100

net4 = VGG('VGG16')
mlps = [net4.to(device)]
optimizer = torch.optim.Adam([{"params": mlp.parameters()} for mlp in mlps], lr=LR)
loss_function = nn.CrossEntropyLoss()
```

**4、迭代优化**

通过定义的训练次数进行模型的参数优化过程，每一次训练输出模型的测试正确率。

```python
for ep in range(EPOCHES):
    for img, label in trainloader:
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        for mlp in mlps:
            mlp.train()
            out = mlp(img)
            loss = loss_function(out, label)
            loss.backward()
        optimizer.step()
    
    pre = []
    vote_correct = 0
    mlps_correct = [0 for i in range(len(mlps))]
    for img, label in testloader:
        img, label = img.to(device), label.to(device)
        for i, mlp in enumerate(mlps):
            mlp.eval()
            out = mlp(img)
            _, prediction = torch.max(out, 1)
            pre_num = prediction.cpu().numpy()
            mlps_correct[i] += (pre_num == label.cpu().numpy()).sum()
            pre.append(pre_num)
        arr = np.array(pre)
        pre.clear()
        result = [Counter(arr[:, i]).most_common(1)[0][0] for i in range(BATCHSIZE)]
        vote_correct += (result == label.cpu().numpy()).sum()
    
    for idx, correct in enumerate(mlps_correct):
        print("Epoch：" + str(ep) + "VGG的正确率为：" + str(correct/len(testloader)))
        
```

训练输出如下所示：


当 Epoch：40    LR：0.02

输出如下：

![image-20211118013658688](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20211118013658688.png)

可以观察到，随着模型迭代次数增多，正确率逐渐在一个值附近波动。最高正确率达88.97%



当Epoch： 40    LR： 0.1 

输出如下：

可以观察到，由于学习率特别高，导致损失变大，正确率反而降低。
