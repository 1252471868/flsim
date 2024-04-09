import load_data
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Training settings
lr = 0.001
momentum = 0.5
log_interval = 10

# Cuda settings
use_cuda = torch.cuda.is_available()
device = torch.device (  # pylint: disable=no-member
    'cuda' if use_cuda else 'cpu')


class Generator(load_data.Generator):
    """Generator for FashionMNIST dataset."""

    # Extract FashionMNIST data using torchvision datasets
    def read(self, path):
        image_size = 28
        data_transform = transforms.Compose([
            # transforms.ToPILImage(),  
            # 这一步取决于后续的数据读取方式，如果使用内置数据集读取方式则不需要
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        self.trainset = datasets.FashionMNIST(
            path, train=True, download=True, transform=data_transform)
        self.testset = datasets.FashionMNIST(
            path, train=False, transform=data_transform)
        labels = list(self.trainset.classes)
        self.labels = [index for index, _ in enumerate(labels)]


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))
#         self.fc = nn.Linear(7 * 7 * 32, 10)

#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = out.reshape(out.size(0), -1)
#         out = self.fc(out)
#         return out
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*4*4, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64*4*4)
        x = self.fc(x)
        # x = nn.functional.normalize(x)
        return x

def get_optimizer(model):
    return optim.Adam(model.parameters(), lr=lr)


def get_trainloader(trainset, batch_size):
    return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)


def get_testloader(testset, batch_size):
    return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


def extract_weights(model):
    weights = []
    for name, weight in model.to(torch.device('cpu')).named_parameters():  # pylint: disable=no-member
        if weight.requires_grad:
            weights.append((name, weight.data))

    return weights


def load_weights(model, weights):
    updated_state_dict = {}
    for name, weight in weights:
        updated_state_dict[name] = weight

    model.load_state_dict(updated_state_dict, strict=False)

def extract_weights_noname(model):
    weights = []
    for weight in model.to(torch.device('cpu')).parameters():  # pylint: disable=no-member
        if weight.requires_grad:
            weights.append((weight.data))
        # print(weight.size())
    return weights

def load_weights_noname(model, weights):
    updated_state_dict = {}
    idx=0
    for name, _ in model.named_parameters():
        updated_state_dict[name] = weights[idx]
        idx=idx+1
    model.load_state_dict(updated_state_dict, strict=False)

def train(model, trainloader, optimizer, epochs):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_batches = 0
    for epoch in range(1, epochs + 1):
        for batch_id, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1
            if batch_id % log_interval == 0:
                logging.debug('Epoch: [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, epochs, loss.item()))
    average_loss = total_loss / total_batches
    return average_loss

def test(model, testloader):
    model.to(device)
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(  # pylint: disable=no-member
                outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    logging.debug('Accuracy: {:.2f}%'.format(100 * accuracy))

    return accuracy
