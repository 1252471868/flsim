import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
df1 = pd.read_csv('output/mnist_fedavg_noniid.csv')
df2 = pd.read_csv('output/mnist_fedavg_iid.csv')


# Plot data using accuracy vs. round
fig, ax = plt.subplots()
sns.lineplot(data=df1, x='round', y='accuracy', label='FedAvg (IID)', ax=ax)
sns.lineplot(data=df2, x='round', y='accuracy', label='FedAvg (IID)', ax=ax)

#ax.set_title('MNIST')
ax.set_xlabel('Communication Round')
ax.set_ylabel('Testing Accuracy (%)')
# turn on grid lines
ax.grid(True)
ax.set_ylim(60, 100)
ax.set_xlim(0, 200)
#plt.show()

# save the figure as a PNG
fig.savefig('output/fig_1.png')