import pickle
from server import Server
import numpy as np
import torch
from agent.agent import Agents
from agent.replay_buffer import ReplayBuffer
import logging
from datetime import datetime

class MARLTrainServer(Server):
	"""Multi-Agent server that performs accuracy weighted federated averaging."""
	def __init__(self, config, case_name):
		super().__init__(config,case_name)

		self.n_episodes = config.marl.n_episodes
		self.n_steps = config.marl.n_steps
		self.n_eval_steps = config.marl.n_eval_steps
		self.n_actions = config.marl.n_actions
		self.n_agents = config.marl.n_agents
		# self.obs_shape = config.obs_shape
		self.epsilon_anneal_scale = config.marl.epsilon_anneal_scale
		self.epsilon = config.marl.epsilon
		self.anneal_epsilon = config.marl.anneal_epsilon
		self.min_epsilon = config.marl.min_epsilon

		self.w1 = 1
		self.w2 = 0.2
		self.w3 = 0.1
		self.h_size=3
		self.probing_losses = np.zeros(self.n_agents)
		self.probing_latencies = np.zeros(self.n_agents)
		self.rest_training_latencies = np.zeros(self.n_agents)
		self.comm_latencies = np.zeros(self.n_agents)
		self.comm_costs = np.zeros(self.n_agents)
		self.data_sizes = np.zeros(self.n_agents)
		self.round_index = 0
		self.acc = 0
		self.acc_last = 0
		# state = np.hstack([self.probing_losses, self.probing_latencies,
		# 			  self.comm_latencies, self.comm_costs, self.data_sizes])
		self.state_shape = 5

		self.agents = Agents(config, self.state_shape)
		self.buffer = ReplayBuffer(config, self.state_shape)

	
	def U(self, x):
		return 10-20/(1+np.exp(0.35*(1-x)))

	def normalization(self, xs):
		xmin = min(xs)
		xmax = max(xs)
		if xmax == xmin:
			return np.ones(self.n_agents)
		normalized = [(x - xmin)/(xmax - xmin) for x in xs]
		return normalized

	def step(self, actions):
		indices  = np.where(np.array(actions)==1)[0]
		if indices.size != 0:
			# return -999
			self.acc, self.comm_latencies, self.rest_training_latencies = self.round(actions)
		else:
			self.acc = 0
			self.comm_latencies = np.full(self.n_agents, 31)
			self.rest_training_latencies = np.full(self.n_agents, 1.2)
		# max_sum = max(a + b for a, b in zip(self.normalization(self.comm_latencies), self.normalization(self.rest_training_latencies)))
		max_sum = max(a + b for a, b in zip(self.comm_latencies / 1441, self.rest_training_latencies / 51))
		# H_t=max(self.normalization(self.probing_latencies))+max_sum 
		H_t=max(self.probing_latencies / 51)+max_sum 
		reward = - self.w1*(self.U(self.acc)-self.U(self.acc_last))-self.w2*H_t-self.w3*max(self.comm_costs/45)
		self.acc_last = self.acc

		return reward


	def round(self, actions):
		import fl_model  # pylint: disable=import-error
		indices  = np.where(np.array(actions)==1)[0]
		if indices.size == 0:
			return 0, np.ones(self.n_agents), np.ones(self.n_agents)
		self.round_index=self.round_index+1
		sample_clients=[self.clients[i] for i in indices]
		sample_clients_id = [client.client_id for client in sample_clients]
		self.configuration(sample_clients)
		
		# Wait for reports
		# self.socket.listen_K_clients(len(sample_clients))
		while not all(self.socket.clients_list.get_report_state(sample_clients_id)):
			pass
		self.socket.clients_list.clear_report_state()
		# Run clients using multithreading for better parallelism

		# Recieve client updates
		reports = self.reporting(sample_clients)
		comm_latencies = np.zeros(self.n_agents)
		rest_training_latencies = np.zeros(self.n_agents)
		for i,report in zip(sample_clients_id, reports):
			time_diff = datetime.now() - report.comm_latency
			report.comm_latency=time_diff.total_seconds()
			comm_latencies[i] = report.comm_latency
			rest_training_latencies[i] = report.training_latency
		# Perform weight aggregation
		logging.info('Aggregating updates')
		updated_weights = self.aggregation(reports)

		# Load updated weights
		fl_model.load_weights(self.model, updated_weights)
		# fl_model.load_weights_noname(self.model, weights_unflattened)

		# Extract flattened weights (if applicable)
		# if self.config.paths.reports:
		# 	self.save_reports(round, reports)

		# Save updated global model
		self.save_model(self.model, self.config.paths.model)

		# Test global model accuracy
		if self.config.clients.do_test:  # Get average accuracy from client reports
			accuracy = self.accuracy_averaging(reports)
		else:  # Test updated model on server
			logging.info('Testing...')
			testset = self.loader.get_testset()
			batch_size = self.config.fl.batch_size
			testloader = fl_model.get_testloader(testset, batch_size)
			accuracy = fl_model.test(self.model, testloader)

		logging.info('Average accuracy: {:.2f}%\n'.format(100 * accuracy))
		with open(self.config.marl.model_dir+'/'+'accuracy'+'.csv', 'a') as f:
				f.write('{},{:.4f}'.format(self.round_index, accuracy*100)+'\n')
		return accuracy, comm_latencies, rest_training_latencies

	def reset(self):
		self.probing_losses = np.zeros(self.n_agents)
		self.probing_latencies = np.zeros(self.n_agents)
		self.rest_training_latency = np.zeros(self.n_agents)
		self.comm_latency = np.zeros(self.n_agents)
		self.comm_cost = np.zeros(self.n_agents)
		self.data_size = np.zeros(self.n_agents)
		# self.round_index = 0;    
		

	def generate_episode(self, episode_num=None, evaluate=False):
		u, r, s, u_onehot, terminate, padded = [], [], [], [], [], []
		self.reset()
		terminated = False
		# finish_tag = False
		step = 0
		episode_reward = 0  # cumulative rewards
		# last_action = np.zeros((self.n_agents, self.n_actions))
		self.agents.policy.init_hidden(1)

		# epsilon
		epsilon = 0 if evaluate else self.epsilon
		if self.epsilon_anneal_scale == 'episode':
			epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

		while step < self.n_steps:
			# time.sleep(0.2)
			# obs = self.env.get_obs()
			self.probing_losses, self.probing_latencies, self.comm_latencies = self.probing_training()
			state = self.get_state()
			actions, avail_actions, actions_onehot = [], [], []
			for agent_id in range(self.n_agents):
				# avail_action = self.env.get_avail_agent_actions(agent_id)
				action = self.agents.choose_action(state[agent_id], agent_id, epsilon)
				# generate onehot vector of th action
				action_onehot = np.zeros(self.n_actions)
				action_onehot[action] = 1
				actions.append(action)
				actions_onehot.append(action_onehot)
				# avail_actions.append(avail_action)
				# last_action[agent_id] = action_onehot

			reward = self.step(actions)
			# finish_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False
			# o.append(obs)
			s.append(state)
			u.append(np.reshape(actions, [self.n_agents, 1]))
			u_onehot.append(actions_onehot)
			# avail_u.append(avail_actions)
			r.append([reward])
			# terminate.append([terminated])
			# padded.append([0.])
			episode_reward += reward
			step += 1
			if self.epsilon_anneal_scale == 'step':
				epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
		# last obs
		# obs = self.env.get_obs()
		state = self.get_state()
		# o.append(obs)
		s.append(state)
		# o_next = o[1:]
		s_next = s[1:]
		# o = o[:-1]
		s = s[:-1]
		# get avail_action for last obs，because target_q needs avail_action in training

		# if step < self.episode_limit，padding
		for i in range(step, self.n_steps):
			u.append(np.zeros([self.n_agents, 1]))
			s.append(np.zeros(self.state_shape))
			r.append([0.])
			s_next.append(np.zeros(self.state_shape))
			u_onehot.append(np.zeros((self.n_agents, self.n_actions)))

		episode = dict(s=s.copy(),
					   u=u.copy(),
					   r=r.copy(),
					   s_next=s_next.copy(),
					   u_onehot=u_onehot.copy(),
					   )
		# add episode dim
		for key in episode.keys():
			episode[key] = np.array(episode[key])

		if not evaluate:
			self.epsilon = epsilon
		return episode, episode_reward, step
	
	def run(self):
		if not self.config.marl.load_model:
			i_episode, train_steps, evaluate_steps = 0, 0, -1
			with open(self.config.marl.model_dir+'/'+'accuracy'+'.csv', 'w') as f:
				f.write('round,accuracy\n')
			while i_episode < self.config.marl.n_episodes:
				# 收集self.config.n_episodes个episodes
				logging.info('-------- Episode: {} --------%\n'.format(i_episode))
				episode, _, steps = self.generate_episode()
				with open(self.config.marl.model_dir+'/'+'state'+'.pkl', 'ab') as f:
					pickle.dump(episode, f)
				i_episode += 1
				# episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs拼在一起
				self.buffer.store_episode(episode)
				for train_step in range(1, self.config.marl.train_steps+1):
					mini_batch = self.buffer.sample(min(self.buffer.current_size, self.config.marl.batch_size))
					self.agents.train(mini_batch, train_step)
		else:
			logging.info('Evaluating......\n')
			accuracy =  self.evaluate()
			logging.info('Evaluated average accuracy: {:.2f}%\n'.format(100 * accuracy))

		return
	
	def probing_training(self):
		import fl_model  # pylint: disable=import-error
		clients = self.clients
		# Configure clients to train on local data
		logging.info('Probing training...')
		self.configuration(clients, probing_training=True)     
		sample_clients_ids = [client.client_id for client in clients]   
		# Train on local data for profiling purposes
		# Wait for reports
		while not all(super().get_clientlist().get_report_state(sample_clients_ids)):
			pass
		self.socket.clients_list.clear_report_state()
		# Recieve client reports
		reports = self.reporting(clients)
		probing_losses=[]
		probing_latencies=[]
		comm_latencies = []
		for report in reports:
			time_diff = datetime.now() - report.comm_latency
			report.comm_latency=time_diff.total_seconds()
			comm_latencies.append(report.comm_latency)
			probing_losses.append(report.loss)
			probing_latencies.append(report.training_latency)
		# Perform weight aggregation
		logging.info('Probing training completed')
		return np.array(probing_losses), np.array(probing_latencies), np.array(comm_latencies)
		 

	def evaluate(self):
		self.reset()
		step = 0
		self.agents.policy.init_hidden(1)
		# epsilon
		epsilon = 0
		acc = []
		with open(self.config.marl.model_dir+'/'+'eval_accuracy'+'.csv', 'w') as f:
			f.write('round, accuracy\n')
		while step < self.n_eval_steps:
			step=step+1
			logging.info('Round: {}\n'.format(step))
			self.probing_losses, self.probing_latencies, self.comm_latencies = self.probing_training()
			state = self.get_state()
			actions, avail_actions, actions_onehot = [], [], []
			for agent_id in range(self.n_agents):
				action = self.agents.choose_action(state[agent_id], agent_id, epsilon)
				action_onehot = np.zeros(self.n_actions)
				action_onehot[action] = 1
				actions.append(action)
				actions_onehot.append(action_onehot)
			indices  = np.where(np.array(actions)==1)[0]
			if indices.size != 0:
				self.acc, self.comm_latencies, self.rest_training_latencies = self.round(actions)
				acc.append(self.acc)

				with open(self.config.marl.model_dir+'/'+'eval_accuracy'+'.csv', 'a') as f:
					f.write('{},{:.4f}'.format(step, self.acc*100)+'\n')
		if len(acc) != 0:
			accuracy = sum(acc) / len(acc)
			return accuracy
		return 0
	
	def select_top_k(self, losses):
		top_k_index = np.argsort(losses)[-self.n_agents:]
		
		sample_clients = [self.clients[idx] for idx in top_k_index]
		print("top_k_index: ", top_k_index)
		return sample_clients
		
		 

	def get_state_agent(self, agent_id):
		#TODO
		return [self.probing_losses[agent_id], 
		  			self.rest_training_latencies[agent_id], 
		  			self.comm_latencies[agent_id],
					self.comm_costs[agent_id],
					self.data_sizes[agent_id]
					# self.round_index
		  			]

	
	def get_state(self):
		"""Returns all agent observations in a list.
		NOTE: Agents should have access only to their local observations
		during decentralised execution.
		"""
		agents_state = [self.get_state_agent(i) for i in range(self.n_agents)]
		return agents_state

	def get_probing_loss(self):
		return self.probing_losses
	
	def get_training_latency(self):
		return self.rest_training_latency
	
	def get_comm_latency(self):
		return
	
	def get_comm_cost(self):
		return
	
	def get_data_size(self):
		return
	
	def get_round_index(self):
		return
	


class MARLServer(MARLTrainServer):
	"""
	FL server using pre-trained MARL  to select top k devices based on the MARL's output
	"""
	
	def __init__(self, config, case_name):
		
		super().__init__(config,case_name)

	
	def load_pca(self, pca_model_fn):
		print("Load saved PCA model from:", pca_model_fn)
		self.pca = pk.load(open(pca_model_fn,'rb'))
		print("PCA model loaded.")        

	def load_dqn_model(self, trained_model):
		self.dqn_model = keras.models.load_model(trained_model)
		print("Loaded trained DQN model from:", trained_model)


	# Set up server
	def boot(self):
		logging.info('Booting {} server...'.format(self.config.server))

		model_path = self.config.paths.model
		total_clients = self.config.clients.total
		client_cfg = self.config.clients
		# Add fl_model to import path
		sys.path.append(model_path)

		# Set up simulated server
		self.load_data()
		self.load_model() # save initial global model
		# self.make_clients(total_clients)
		self.connect_clients(client_cfg)
		# load PCA model and pretrained DQN model
		self.load_pca(self.config.dqn.pca_model)
		self.load_dqn_model(self.config.dqn.trained_model)
	
	# Run federated learning with multiple communication round, each round the participating devices
	# are selected by the trained dqn agent given the current state
	def run(self):

		rounds = self.config.fl.rounds
		target_accuracy = self.config.fl.target_accuracy
		reports_path = self.config.paths.reports

		if target_accuracy:
			logging.info('Training: {} rounds or {}% accuracy\n'.format(
				rounds, 100 * target_accuracy))
		else:
			logging.info('Training: {} rounds\n'.format(rounds))
		
		with open('output/'+self.case_name+'.csv', 'w') as f:
			f.write('round,accuracy\n')

		# initial check in with server, all clients send their initial weights to server
		self.profile_all_clients(train_dqn=False)

		# Perform rounds of federated learning
		for round in range(1, rounds + 1):
			logging.info('**** Round {}/{} ****'.format(round, rounds))
			accuracy = self.round()

			with open('output/'+self.case_name+'.csv', 'a') as f:
				f.write('{},{:.4f}'.format(round, accuracy*100)+'\n')

			# Break loop when target accuracy is met
			if target_accuracy and (accuracy >= target_accuracy):
				logging.info('Target accuracy reached.')
				break

		if reports_path:
			with open(reports_path, 'wb') as f:
				pk.dump(self.saved_reports, f)
			logging.info('Saved reports: {}'.format(reports_path))


	# override the round() method in the server with dqn_selection() based on observed states
	def round(self):

		import fl_model  # pylint: disable=import-error

		# Select clients to participate in the round
		sample_clients = self.dqn_select_top_k()
		sample_clients_ids = [client.client_id for client in sample_clients] 
		print("sample_clients_ids: ", sample_clients_ids)

		# Configure sample clients
		self.configuration(sample_clients)

		# Run clients using multithreading for better parallelism
		# Wait for reports
		while not all(super().get_clientlist().get_report_state(sample_clients_ids)):
			pass
		super().get_clientlist().clear_report_state()
		# Receive client updates
		reports = self.reporting(sample_clients)

		# update the pca weights for each client
		clients_weights = [self.flatten_weights(report.weights) for report in reports] # list of numpy arrays
		clients_weights = np.array(clients_weights) # convert to numpy array
		clients_weights_pca = self.pca.transform(clients_weights)

		# Perform weight aggregation
		logging.info('Aggregating updates')
		updated_weights = self.aggregation(reports)

		# update the pca weights for the server
		server_weights = [self.flatten_weights(updated_weights)]
		server_weights = np.array(server_weights)
		server_weights_pca = self.pca.transform(server_weights)

		# update the weights of the selected devices and server to corresponding client id 
		# return next_state
		for i in range(len(sample_clients_ids)):
			self.pca_weights_clientserver[sample_clients_ids[i]] = clients_weights_pca[i]
		
		self.pca_weights_clientserver[-1] = server_weights_pca[0]

		# Load updated weights
		fl_model.load_weights(self.model, updated_weights)

		# Extract flattened weights (if applicable)
		if self.config.paths.reports:
			self.save_reports(round, reports)

		# Save updated global model
		self.save_model(self.model, self.config.paths.model)

		# Test global model accuracy
		if self.config.clients.do_test:  # Get average test accuracy from client reports
			print('Get average accuracy from client reports')
			accuracy = self.accuracy_averaging(reports)

		else:  # Test updated model on server using the aggregated weights
			print('Test updated model on server')
			testset = self.loader.get_testset()
			batch_size = self.config.fl.batch_size
			testloader = fl_model.get_testloader(testset, batch_size)
			accuracy = fl_model.test(self.model, testloader)

		logging.info('Average accuracy: {:.2f}%\n'.format(100 * accuracy))

		return accuracy # this is testing accuracy  


	def dqn_select_top_k(self):
		
		# Select devices to participate in current round
		clients_per_round = self.config.clients.per_round
		print('self.pca_weights_clientserver.shape:', self.pca_weights_clientserver.shape)

		# calculate state using the pca model transformed weights
		state = self.pca_weights_clientserver.flatten()
		state = state.tolist()

		# use dqn model to select top k devices
		q_values = self.dqn_model.predict([state])[0]
		print("q_values: ", q_values)

		# select top k index based on the q_values
		top_k_index = np.argsort(q_values)[-clients_per_round:]
		print("top_k_index: ", top_k_index)

		sample_clients = [self.clients[idx] for idx in top_k_index]

		return sample_clients



