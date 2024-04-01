import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

num='4'

data = []
with open('output/marl_models/' + num +'/state.pkl', 'rb') as pickle_file:
    while True:
        try:
            data.append(pickle.load(pickle_file))
        except EOFError:
            break
# Load data
# df1 = pd.read_csv('output/fig1/mnist_fedavg_iid.csv')
# df0 = pd.read_csv('output/marl_models/' + num + '/accuracy.csv')
df1 = pd.read_csv('output/marl_models/' + num + '/mnist_fedavg_noniid.csv')
df2 = pd.read_csv('output/marl_models/' + num + '/eval_accuracy.csv')
episode = data[-15:]
state = episode[0]['s']
action = episode[0]['u']
reward = episode[0]['r']
for i in range(1,15):
    state = np.vstack((state, episode[i]['s']))
    action = np.vstack((action, episode[i]['u']))
    reward = np.vstack((reward, episode[i]['r']))
probing_loss = state[:,:,0]
process_latency = state[:,:,1]
comm_latency = state[:,:,2]

proc_lan=[]
comm_lan=[]
proc_lan.append(np.sum(process_latency[:,0]))
proc_lan.append(np.sum(process_latency[:,1]))
proc_lan.append(np.sum(process_latency[:,2]))
comm_lan.append(np.sum(comm_latency[:,0]))
comm_lan.append(np.sum(comm_latency[:,1]))
comm_lan.append(np.sum(comm_latency[:,2]))

# Plot data using accuracy vs. round

# x = np.linspace(3,45,15)
x_ind = range(2, 45, 3)
x = range(15)
y_loss = probing_loss[x_ind,:]
y_process_latency = process_latency[x_ind,:]
y_comm_latency = comm_latency[x_ind,:]
act = action[x_ind,:]
loss_selected=np.full((15, 3), np.nan)
loss_unselected=np.full((15, 3), np.nan)
process_latency_selected=np.full((15, 3), np.nan)
process_latency_unselected=np.full((15, 3), np.nan)
comm_latency_selected=np.full((15, 3), np.nan)
comm_latency_unselected=np.full((15, 3), np.nan)
for i in range(15):
    for j, a in enumerate(act[i]):
        if a == 1:
            loss_selected[i][j]=y_loss[i][j]
            process_latency_selected[i][j]=y_process_latency[i][j]
            comm_latency_selected[i][j]=y_comm_latency[i][j]
        else:
            loss_unselected[i][j]=y_loss[i][j]
            process_latency_unselected[i][j]=y_process_latency[i][j]
            comm_latency_unselected[i][j]=y_comm_latency[i][j]
fig, ax = plt.subplots()
for i in range(loss_selected.shape[1]):
    plt.scatter(x, loss_selected[:,i], color='blue', label='Selected')  # Blue points
    plt.scatter(x, loss_unselected[:,i], color='red', label='Not Selected')  # Red points
# Labeling
plt.xlabel('Probing loss')
plt.ylabel('Episode')
plt.legend()
# Show plot
plt.show()
# save the figure as a PNG
fig.savefig('output/marl_models/probing_loss.png')


fig, ax = plt.subplots()
for i in range(loss_selected.shape[1]):
    plt.scatter(x, process_latency_selected[:,i], color='blue', label='Selected')  # Blue points
    plt.scatter(x, process_latency_unselected[:,i], color='red', label='Not Selected')  # Red points
# Labeling
# plt.title('probing loss')
plt.xlabel('Processing latency')
plt.ylabel('Episode')
plt.legend()
# Show plot
plt.show()
# save the figure as a PNG
fig.savefig('output/marl_models/process_latency.png')


fig, ax = plt.subplots()
for i in range(loss_selected.shape[1]):
    plt.scatter(x, comm_latency_selected[:,i], color='blue', label='Selected')  # Blue points
    plt.scatter(x, comm_latency_unselected[:,i], color='red', label='Not Selected')  # Red points
# Labeling
# plt.title('probing loss')
plt.xlabel('Comm latency')
plt.ylabel('Episode')
plt.legend()
# Show plot
plt.show()
# save the figure as a PNG
fig.savefig('output/marl_models/comm_latency.png')



fig, ax = plt.subplots()
# sns.lineplot(data=df1, x='round', y='accuracy', label='FedAvg (IID)', ax=ax)
sns.lineplot(data=df2, x='round', y='accuracy', label='MARL (Non-IID), UDP, 3 Clients', ax=ax)
sns.lineplot(data=df1, x='round', y='accuracy', label='FedAvg (Non-IID), UDP, 3 Clients', ax=ax)
ax.set_title('Wireless Communication ')
ax.set_xlabel('Communication Round')
ax.set_ylabel('Testing Accuracy (%)')
# turn on grid lines
ax.grid(True)
ax.set_ylim(60, 100)
ax.set_xlim(0, 150)
#plt.show()

# save the figure as a PNG
fig.savefig('output/marl_models/accuracy.png')


fig, ax = plt.subplots()
x_reward = range(45)
plt.plot(x_reward, reward)
# Labeling
# plt.title('probing loss')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
# Show plot
plt.show()
# save the figure as a PNG
fig.savefig('output/marl_models/reward.png')