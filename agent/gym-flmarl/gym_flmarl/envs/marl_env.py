import math
from typing import Optional, Union

import numpy as np
from gym.utils import seeding
import gym
from gym import logger, spaces
from gym.error import DependencyNotInstalled

DDPG_MODE = 1
DQN_MODE = 2
PDQN_MODE = 3
RANDOM_MODE = 4
TD3_MODE = 5
H_SIZE = 10

class FLMARL_Env(gym.Env):
    """
    ### Description
    
    """

    def __init__(self, config):
        self.seed()
        self.w1 = 1
        self.w2 = 0.2
        self.w3 = 0.1
        self.n_agents = config.clients.total
        self.h_size = H_SIZE;
        self.probing_loss = np.zeros(self.n_agents)
        self.probing_training_latency = np.zeros((self.n_agents, self.h_size))
        self.rest_training_latency = np.zeros((self.n_agents, self.h_size))
        self.comm_latency = np.zeros((self.n_agents, self.h_size))
        self.comm_cost = np.zeros(self.n_agents)
        self.data_size = np.zeros(self.n_agents)
        self.round_index = 0;    
        

    def seed(self, seed=123):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def U(self, x):
        return 10-20/(1+np.exp(0.35*(1-x)))

    def step(self, action):
        acc=0
        acc_1=0
        
        H_t=max(self.probing_training_latency)+max(self.comm_latency+self.rest_training_latency)
        reward = self.w1*(self.U(acc)-self.U(acc_1))-self.w2*H_t-self.w3*self.comm_cost

        return 0

    def reset(self):
        
        return np.array(self.state)

    def get_state_agent(self, agent_id):
        return 
    
    def get_state(self):
        """Returns all agent observations in a list.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        agents_state = [self.get_state_agent(i) for i in range(self.n_agents)]
        return agents_state

    def get_probing_loss(self):
        return
    
    def get_training_latency(self):
        return
    
    def get_comm_latency(self):
        return
    
    def get_comm_cost(self):
        return
    
    def get_data_size(self):
        return
    
    def get_round_index(self):
        return
    

    def render(self):
        pass

    def close(self):
        pass
