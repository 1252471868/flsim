import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
# df1 = pd.read_csv('output/fig1/mnist_fedavg_iid.csv')
df2 = pd.read_csv('output/fig1/mnist_fedavg_noniid_2_1_udp_0loss_IPC.csv')
# df3 = pd.read_csv('output/fig1/mnist_kcenter_noniid.csv')
# df4 = pd.read_csv('output/fig1/mnist_fedavg_noniid_tcp.csv')
df4 = pd.read_csv('output/fig1/mnist_fedavg_noniid_2_1_udp_5loss_IPC.csv')
df5 = pd.read_csv('output/fig1/mnist_fedavg_noniid_2_1_udp_10loss_IPC.csv')
df6 = pd.read_csv('output/fig1/mnist_fedavg_noniid_2_1_udp_15loss_IPC.csv')
df7 = pd.read_csv('output/fig1/mnist_fedavg_noniid_2_1_udp_20loss_IPC.csv')

# Plot data using accuracy vs. round
fig, ax = plt.subplots()
# sns.lineplot(data=df1, x='round', y='accuracy', label='FedAvg (IID)', ax=ax)
sns.lineplot(data=df2, x='round', y='accuracy', label='FedAvg (Non-IID), UDP, 2 Clients', ax=ax)
# sns.lineplot(data=df3, x='round', y='accuracy', label='FedAvg (Non-IID), UDP, 1 Client', ax=ax)
sns.lineplot(data=df4, x='round', y='accuracy', label='FedAvg (Non-IID), UDP, 2 Clients, 5%', ax=ax)
sns.lineplot(data=df5, x='round', y='accuracy', label='FedAvg (Non-IID), UDP, 2 Clients, 10%', ax=ax)
sns.lineplot(data=df6, x='round', y='accuracy', label='FedAvg (Non-IID), UDP, 2 Clients, 15%', ax=ax)
sns.lineplot(data=df7, x='round', y='accuracy', label='FedAvg (Non-IID), UDP, 2 Clients, 20%', ax=ax)

 
ax.set_title('Wireless Communication ')
ax.set_xlabel('Communication Round')
ax.set_ylabel('Testing Accuracy (%)')
# turn on grid lines
ax.grid(True)
ax.set_ylim(60, 100)
ax.set_xlim(0, 150)
#plt.show()

# save the figure as a PNG
fig.savefig('output/fig1/fig_1.png')