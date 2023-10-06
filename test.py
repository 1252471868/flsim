import torch
import torch.nn as nn
import sys
import torchvision



# 检查是否有可用的GPU
if torch.cuda.is_available():
    print(torch.version.cuda)
    device = torch.device("cuda:0")  # 使用第一个GPU
else:
    device = torch.device("cpu")
    print('cpu')
# device = torch.device("cpu")

# 定义一个简单的神经网络模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# 创建模型实例，并将其移动到GPU上
model = SimpleModel().to(device)

# 定义输入数据
inputs = torch.tensor([[1.0], [2.0], [3.0], [4.0]]).to(device)

# 在多个GPU上进行并行计算
outputs = model(inputs)

# 打印计算结果
print(outputs)

print(sys.path)
