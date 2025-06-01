# 使用卷积神经网络
# 将原本3通道100*100的图片，提取特征到10*10以内
import torch.nn as nn


# 定义神经网络模型类
class MyNet(nn.Module):
    # 构造函数
    def __init__(self):
        super().__init__()
        # 神经网络设计
        # 创建神经网络序列
        self.seq = nn.Sequential(
            # 第一次卷积，使用10个5*5的卷积核    10*96*96
            nn.Conv2d(3, 10, 5),
            # 归一化、激活函数(优化)
            nn.BatchNorm2d(10),
            nn.ReLU(),
            # 第一次池化，使用2*2窗口移动2个步长  10*48*48
            nn.MaxPool2d(2, 2),
            # 第二次卷积，使用20个5*5的卷积核   20*44*44
            nn.Conv2d(10, 20, 5),
            # 归一化、激活函数(优化)
            nn.BatchNorm2d(20),
            nn.ReLU(),
            # 第二次池化，使用2*2窗口移动2个步长 20*22*22
            nn.MaxPool2d(2, 2),
            # 第三次卷积，使用30个5*5的卷积核  30*18*18
            nn.Conv2d(20, 30, 5),
            # 归一化、激活函数(优化)
            nn.BatchNorm2d(30),
            nn.ReLU(),
            # 第三次池化，使用2*2窗口移动2个步长 30*9*9
            nn.MaxPool2d(2, 2),
            # 正则化(优化)
            nn.Dropout(),
            # 全连接层，展平 30*9*9 300 75 25 3
            nn.Flatten(),
            nn.Linear(30 * 9 * 9, 300),
            nn.Linear(300, 75),
            nn.Linear(75, 25),
            nn.Linear(25, 4),
        )

    # 前向传播
    def forward(self, x):
        return self.seq(x)
