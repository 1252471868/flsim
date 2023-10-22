import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv("dqn_rewards.csv")

episode1 = df1['Episode']
reward1 = df1['Reward']

plt.plot(episode1, reward1)
plt.xlabel('Episode(s)')
plt.ylabel('Total Return')
plt.grid()
plt.title('DQN training with different rewards')
plt.ylim([-20, 80])

plt.show()
plt.savefig("dqn_rewards.png")

