from numpy import size
import numpy as np
import load_data
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import io
import dill
import sys

# Training settings
lr = 0.01
momentum = 0.5
log_interval = 10

# Cuda settings
use_cuda = torch.cuda.is_available()
device = torch.device(  # pylint: disable=no-member
    'cuda' if use_cuda else 'cpu')



class Report(object):
	"""Federated learning client report."""

	def __init__(self, id = 0, num_sample = 0):
		self.weights = []
		self.client_id = id
		self.num_samples = num_sample
  
  
class TensorBuffer:
    """
    Class to flatten and deflatten the gradient vector.
    """

    def __init__(self, tensors):
        indices = [0]
        for tensor in tensors:
            new_end = indices[-1] + tensor.nelement()
            indices.append(new_end)

        self._start_idx = indices[:-1]
        self._end_idx = indices[1:]
        self._len_tensors = len(tensors)
        self._tensor_shapes = [tensor.size() for tensor in tensors]

        self.buffer = torch.cat([tensor.view(-1) for tensor in tensors])

    def __getitem__(self, index):
        return self.buffer[self._start_idx[index] : self._end_idx[index]].view(self._tensor_shapes[index])

    def __len__(self):
        return self._len_tensors

def Unflatten_Tensor(flattened_tensors, model):
    buffer = []
    for idx, _ in enumerate(model.parameters()):
        
        buffer.append(flattened_tensors[idx])
    return buffer

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def get_optimizer(model):
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum)


def get_trainloader(trainset, batch_size):
    return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)


def get_testloader(testset, batch_size):
    return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)


# def extract_weights(model):
#     weights = []
#     for name, weight in model.to(torch.device('cpu')).named_parameters():  # pylint: disable=no-member
#         if weight.requires_grad:
#             weights.append((name, weight.data))

#     return torch.tensor(weights)

# def load_weights(model, weights):
#     updated_state_dict = {}
#     for name, weight in weights:
#         updated_state_dict[name] = weight

#     model.load_state_dict(updated_state_dict, strict=False)

def extract_weights(model):
    weights = []
    for weight in model.to(torch.device('cpu')).parameters():  # pylint: disable=no-member
        if weight.requires_grad:
            weights.append((weight.data))
        # print(weight.size())
    return weights

def load_weights(model, weights):
    updated_state_dict = {}
    idx=0
    for name, _ in model.named_parameters():
        updated_state_dict[name] = weights[idx]
        idx=idx+1
    model.load_state_dict(updated_state_dict, strict=False)

def encode(tensor):
    file = io.BytesIO()
    torch.save(tensor, file)

    file.seek(0)
    encoded = file.read()

    return encoded

def decode(buffer):
    tensor = torch.load(io.BytesIO(buffer))
    return tensor

model = Net()
weights = extract_weights(model)
weights_flat = TensorBuffer(weights)


aggregated_weights = torch.zeros(431080)
# print(weights)
weight_split = torch.split(weights_flat.buffer, 100)
print(weights_flat.buffer.size())
# print(weight_split.size())
print(weight_split[0].size())
print(sys.getsizeof(weight_split))
print(sys.getsizeof(weight_split[0]))
weight_np = weight_split[4310].numpy()
weight_np2 = np.concatenate((np.array([100]), weight_np), dtype=np.float32)
weight_encoded_tensor = encode(weight_split[4310].numpy())
# weight_encoded_tensor = encode(weights_flat.buffer)
print('torch save:{}'.format(sys.getsizeof(weight_encoded_tensor)))
weight_encoded_dill = dill.dumps(weight_split[4310].numpy())
# weight_encoded_dill = dill.dumps(weights_flat.buffer.numpy())
print('torch save:{}'.format(sys.getsizeof(weight_encoded_dill)))
buffer = []
for idx, msg in enumerate(weight_split):
    idx_msg = torch.cat([torch.tensor([idx*100]),msg])
    start_idx = idx_msg[0].item()
    start_idx_indices = torch.arange(start_idx, start_idx+len(idx_msg)-1)
    start_index_indices_grad = torch.vstack([start_idx_indices, idx_msg[1:]]).T
    buffer.append(start_index_indices_grad)
    
msg = torch.cat(buffer)
indices = msg[:, 0].long()
weight_recv = msg[:, 1]
print('size: {}'.format())
aggregated_weights[indices] = weight_recv
weights_flat.buffer[:]=aggregated_weights
print(weights_flat[0].size())
weights_unflat = Unflatten_Tensor(weights_flat, model)
load_weights(model,weights_unflat)
weights_extr = extract_weights(model)
sot=np.array(-float("inf"))
sot_encoded = dill.dumps(sot)
print('torch save:{}'.format(sys.getsizeof(sot_encoded)))

report = Report(0,0)
report.weights = weights_flat.buffer.numpy()
report_encoded = dill.dumps(report)
print('torch save:{}'.format(sys.getsizeof(report_encoded)))
print(len(weights_flat))
while(True):
    pass


