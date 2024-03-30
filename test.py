import torch
import torch.nn as nn
import sys
import torchvision
import numpy as np
import gym
import gym_flmarl
import matplotlib.pyplot as plt

ENV_NAME = "FLMARL-v0"

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

a = np.ones(10)
b = np.array(a)
c = range(10)
d = np.asarray([0])
# h = [0]
print(type(d))
ind = np.arange(100,len(d),100)
e = np.array_split(d, ind)



def get_state_agent(agent_id):
    probing_losses = np.zeros(2)
    rest_training_latencies = np.zeros(2)
    # rest_training_latency = np.zeros(2)
    comm_latencies = np.zeros(2)
    comm_costs = np.zeros(2)
    data_sizes = np.zeros(2)
    round_index = 0;   
    return [probing_losses[agent_id], 
                rest_training_latencies[agent_id], 
                comm_latencies[agent_id],
                comm_costs[agent_id],
                data_sizes[agent_id],
                round_index
                ]


def get_state():
    """Returns all agent observations in a list.
    NOTE: Agents should have access only to their local observations
    during decentralised execution.
    """
    agents_state = [get_state_agent(i) for i in range(2)]
    return agents_state

step=0
s=[]
while step<3:
    state=get_state()
    s.append(state)
    step=step+1
s = s[:-1]
u=[3,2,1]
r=[1,2,3]
s_next=[2,3,4]

episode = dict(s=s.copy(),
                # u=u.copy(),
                # r=r.copy(),
            #    o_next=o_next.copy(),
                # s_next=s_next.copy(),
                # u_onehot=u_onehot.copy(),
            #    padded=padded.copy(),
            #    terminated=terminate.copy()
                )
for key in episode.keys():
    episode[key] = np.array(episode[key])
# 打印计算结果
print(outputs)

print(sys.path)

def U(x):
    return 10-20/(1+np.exp(0.35*(1-x)))

x = np.linspace(0,1,100)
y = U(x)
plt.figure()
plt.plot(x,y)
plt.show()
print(U(0.975)-U(0.97))
env = gym.make(ENV_NAME)
