import numpy as np
import torch
from torch.distributions import Categorical
from agent.vdn import VDN

# Agent no communication
class Agents:
    def __init__(self, config, state_shape):
        self.n_actions = config.marl.n_actions
        self.n_agents = config.marl.n_agents
        self.state_shape = state_shape
        # self.obs_shape = config.obs_shape

        self.policy = VDN(config, state_shape)
        self.config = config

    def choose_action(self, obs, agent_num, epsilon):
        inputs = obs.copy()
        # avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose

        # transform agent_num to onehot vector
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.

        # if self.config.last_action:
        #     inputs = np.hstack((inputs, last_action))
        if self.config.marl.reuse_network:
            inputs = np.hstack((inputs, agent_id))
        hidden_state = self.policy.eval_hidden[:, agent_num, :]

        # transform the shape of inputs from (42,) to (1,42)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        # avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        # get q value
        q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)

        # choose action from q value
        # if self.config.alg == 'coma' or self.config.alg == 'central_v' or self.config.alg == 'reinforce':
        #     action = self._choose_action_from_softmax(q_value.cpu(), avail_actions, epsilon)
        # else:
        # q_value[avail_actions == 0.0] = - float("inf")
        if np.random.uniform() < epsilon:
            action = np.random.choice([0,1])  # action从0，1选择
        else:
            action = torch.argmax(q_value).item()
        return action

    def _choose_action_from_softmax(self, inputs, avail_actions, epsilon):
        """
        :param inputs: # q_value of all actions
        """
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # num of avail_actions
        # 先将Actor网络的输出通过softmax转换成概率分布
        prob = torch.nn.functional.softmax(inputs, dim=-1)
        # add noise of epsilon
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0

        """
        不能执行的动作概率为0之后，prob中的概率和不为1，这里不需要进行正则化，因为torch.distributions.Categorical
        会将其进行正则化。要注意在训练的过程中没有用到Categorical，所以训练时取执行的动作对应的概率需要再正则化。
        """
        
        action = Categorical(prob).sample().long()
        return action

    def _get_max_episode_len(self, batch):
        # episode_num = batch['o'].shape[0]
        # episode_num = terminated.shape[0]
        max_episode_len = 0
        # for episode_idx in range(episode_num):
        #     for transition_idx in range(self.config.episode_limit):
        #         if terminated[episode_idx, transition_idx, 0] == 1:
        #             if transition_idx + 1 >= max_episode_len:
        #                 max_episode_len = transition_idx + 1
        #             break
        if max_episode_len == 0:  # 防止所有的episode都没有结束，导致terminated中没有1
            max_episode_len = self.config.marl.n_steps
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):  # coma needs epsilon for training

        # different episode has different length, so we need to get max length of the batch
        # max_episode_len = self._get_max_episode_len(batch)
        max_episode_len = self.config.marl.n_steps
        for key in batch.keys():
            if key != 'z':
                batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.config.marl.save_cycle == 0:
            self.policy.save_model(train_step)









