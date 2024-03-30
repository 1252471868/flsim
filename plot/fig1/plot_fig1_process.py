import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
# df1 = pd.read_csv('output/fig1/mnist_fedavg_iid.csv')
df0 = pd.read_csv('output/fig1/mnist_fedavg_noniid_2_1_udp_0loss_IPC.csv')
# df3 = pd.read_csv('output/fig1/mnist_kcenter_noniid.csv')
# df4 = pd.read_csv('output/fig1/mnist_fedavg_noniid_tcp.csv')
df1 = pd.read_csv('output/fig1/mnist_fedavg_noniid_2_1_udp_5loss_IPC.csv')
df2 = pd.read_csv('output/fig1/mnist_fedavg_noniid_2_1_udp_10loss_IPC.csv')
df3 = pd.read_csv('output/fig1/mnist_fedavg_noniid_2_1_udp_15loss_IPC_3.csv')
df4 = pd.read_csv('output/fig1/mnist_fedavg_noniid_2_1_udp_20loss_IPC.csv')

df_acc = {}
for idx, target in enumerate(range(95, 99)):
    accuracy_exceeding={}
    accuracy_exceeding[0] = df0['accuracy'] > target
    accuracy_exceeding[1] = df1['accuracy'] > target
    accuracy_exceeding[2] = df2['accuracy'] > target
    accuracy_exceeding[3] = df3['accuracy'] > target
    accuracy_exceeding[4] = df4['accuracy'] > target
    first_index_exceeding=[]
    for i in range(5):
        first_index_exceeding.append(accuracy_exceeding[i].idxmax())
    df0_extracted = df0.iloc[:first_index_exceeding[0] + 1]
    df1_extracted = df1.iloc[:first_index_exceeding[1] + 1]
    df2_extracted = df2.iloc[:first_index_exceeding[2] + 1]
    df3_extracted = df3.iloc[:first_index_exceeding[3] + 1]
    df4_extracted = df4.iloc[:first_index_exceeding[4] + 1]
    
    df_acc[idx] = pd.DataFrame({
        'loss': [0, 5, 10, 15, 20],
        'round': first_index_exceeding
    })

# Plot data using accuracy vs. round
fig, ax = plt.subplots()
sns.lineplot(data=df_acc[0], x='loss', y='round', label='Target Accuracy=95%', ax=ax)
sns.lineplot(data=df_acc[1], x='loss', y='round', label='Target Accuracy=96%', ax=ax)
sns.lineplot(data=df_acc[2], x='loss', y='round', label='Target Accuracy=97%', ax=ax)
sns.lineplot(data=df_acc[3], x='loss', y='round', label='Target Accuracy=98%', ax=ax)
# sns.lineplot(data=df1, x='round', y='accuracy', label='FedAvg (IID)', ax=ax)
# sns.lineplot(data=df2, x='round', y='accuracy', label='FedAvg (Non-IID), UDP, 2 Clients', ax=ax)
# sns.lineplot(data=df2, x='round', y='accuracy', label='FedAvg (Non-IID), UDP, 2 Clients', ax=ax)
# sns.lineplot(data=df3, x='round', y='accuracy', label='FedAvg (Non-IID), UDP, 1 Client', ax=ax)
# sns.lineplot(data=df4, x='round', y='accuracy', label='FedAvg (Non-IID), UDP, 2 Clients, 5%', ax=ax)
# sns.lineplot(data=df5, x='round', y='accuracy', label='FedAvg (Non-IID), UDP, 2 Clients, 10%', ax=ax)
# sns.lineplot(data=df6, x='round', y='accuracy', label='FedAvg (Non-IID), UDP, 2 Clients, 15%', ax=ax)
# sns.lineplot(data=df7, x='round', y='accuracy', label='FedAvg (Non-IID), UDP, 2 Clients, 20%', ax=ax)

ax.legend(loc='upper left')
ax.set_title('Wireless Communication ')
ax.set_xlabel('Packet Loss Rate')
ax.set_ylabel('Communication Round')
# turn on grid lines
ax.grid(True)
# ax.set_ylim(60, 100)
# ax.set_xlim(0, 5)
#plt.show()

# save the figure as a PNG
fig.savefig('output/fig1/fig_1_loss.png')