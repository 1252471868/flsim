import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
df1 = pd.read_csv('output/fig1/mnist_fedavg_iid.csv')
df2 = pd.read_csv('output/fig1/mnist_fedavg_noniid_tcp.csv')
# df3 = pd.read_csv('output/fig1/mnist_kcenter_noniid.csv')
df3 = pd.read_csv('output/fig1/mnist_fedavg_noniid_udp.csv')

# Plot data using accuracy vs. round
fig, ax = plt.subplots()
sns.lineplot(data=df1, x='round', y='accuracy', label='FedAvg (IID)', ax=ax)
sns.lineplot(data=df2, x='round', y='accuracy', label='FedAvg (Non-IID), TCP', ax=ax)
sns.lineplot(data=df3, x='round', y='accuracy', label='FedAvg (Non-IID), UDP', ax=ax)

ax.set_title('Wireless Communication, Select 1 out of 2 ')
ax.set_xlabel('Communication Round')
ax.set_ylabel('Testing Accuracy (%)')
# turn on grid lines
ax.grid(True)
ax.set_ylim(60, 100)
ax.set_xlim(0, 100)
#plt.show()

# save the figure as a PNG
fig.savefig('output/fig1/fig_1.png')