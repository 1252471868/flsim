import logging
import math
import random
from server import Server
from sklearn.cluster import KMeans
from threading import Thread
import utils.dists as dists  # pylint: disable=no-name-in-module

from sklearn.decomposition import PCA
import time
from collections import deque
from keras.layers import Dense, Input, Dropout, ReLU
from keras.models import Model
# from tensorflow.
from tensorflow.keras.optimizers import Adam
# from keras.losses import huber_loss
import numpy as np
import pickle as pk
import sys
from tensorflow import keras


class DQNTrainServer(Server):
	"""Federated learning server that uses Double DQN for device selection."""

	def __init__(self, config, case_name):
		
		super().__init__(config,case_name)

		self.memory = deque(maxlen=self.config.dqn.memory_size)
		self.nA = self.config.clients.total
		self.episode = self.config.dqn.episode
		self.max_steps = self.config.dqn.max_steps
		self.target_update = self.config.dqn.target_update
		self.batch_size = self.config.dqn.batch_size
		self.gamma = self.config.dqn.gamma
		# number of components to use for PCA, notice here pca_n_components should be smaller than the total number of clients!!!
		self.pca_n_components = min(100, self.config.clients.total)  
		self.pca = None

		#self.dqn_model = self._build_model()
		#self.target_model = self._build_model()
		
		self.dqn_model = self._build_model2()
		self.target_model = self._build_model2()        

		self.pca_weights_clientserver_init = None
		self.pca_weights_clientserver = None

		print("nA =", self.nA)
		# self.total_steps = 0

	def _build_model(self):
		layers = self.config.dqn.hidden_layers # hidden layers

		# (all clients weight + server weight) * pca_n_components, flattened to 1D
		input_size = (self.config.clients.total + 1) * self.pca_n_components 

		states = Input(shape=(input_size,))
		z = states
		for l in layers:
			z = Dense(l, activation='linear')(z)

		q = Dense(self.config.clients.total, activation='linear')(z) # here use linear activation function to predict the q values for each action/client

		model = Model(inputs=[states], outputs=[q])
		model.compile(optimizer=Adam(learning_rate=self.config.dqn.learning_rate), loss='huber_loss')

		return model

	def _build_model2(self):

		# use the 2layer MLP torch model in fl-lottery/rl/agent.py
		# https://github.com/iQua/fl-lottery/blob/360d9c2d54c12e2631ac123a4dd5ac9184d913f0/rl/agent.py

		layers = self.config.dqn.hidden_layers # hidden layers
		l1 = layers[0]
		l2 = layers[1]

		# (all clients weight + server weight) * pca_n_components, flattened to 1D
		input_size = (self.config.clients.total + 1) * self.pca_n_components 

		states = Input(shape=(input_size,))

		z = Dense(l1, activation='linear')(states)
		z = Dropout(0.5)(z)
		z = ReLU()(z)
		q = Dense(self.config.clients.total, activation='linear')(z)

		model = Model(inputs=[states], outputs=[q])
		model.compile(optimizer=Adam(learning_rate=self.config.dqn.learning_rate), loss='huber_loss')

		return model

	def update_target_model(self):
		# copy weights from model to target_model
		self.target_model.set_weights(self.dqn_model.get_weights())

	def epsilon_greedy(self, state, epsilon_current):

		nA = self.nA
		epsilon = epsilon_current #  the probability of choosing a random action
		action_probs = np.ones(nA, dtype=float) * epsilon / nA
		best_action = np.argmax(self.dqn_model.predict([state])[0])
		action_probs[best_action] += (1 - epsilon)
		action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

		return action

	def memorize(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))


	def create_greedy_policy(self):

		def policy_fn(state):
			return np.argmax(self.dqn_model.predict([state])[0])

		return policy_fn
	

	def dqn_round(self, random=False, action=0):
		# default: select the 

		import fl_model  # pylint: disable=import-error

		# Select clients to participate in the round
		if random:
			sample_clients = self.selection()
			print("randomly select clients:", [client.client_id for client in sample_clients])
		else:
			sample_clients = self.dqn_selection(action)
			print("dqn select clients:", [client.client_id for client in sample_clients]   )

		sample_clients_ids = [client.client_id for client in sample_clients]   

		# Configure sample clients
		self.configuration(sample_clients)

		# Wait for reports
		while not all(super().get_clientlist().get_report_state(sample_clients_ids)):
			pass
		# Run clients using multithreading for better parallelism
		# threads = [Thread(target=client.run) for client in sample_clients]
		# [t.start() for t in threads]
		# [t.join() for t in threads]

		# Receive client updates
		reports = self.reporting(sample_clients) # list of weight tensors

		# client weights pca
		clients_weights = [self.flatten_weights(report.weights) for report in reports] # list of numpy arrays
		clients_weights = np.array(clients_weights) # convert to numpy array
		clients_weights_pca = self.pca.transform(clients_weights)

		# Perform weight aggregation
		logging.info('Aggregating updates')
		updated_weights = self.aggregation(reports)
		# Load updated weights
		fl_model.load_weights(self.model, updated_weights)

		# save the updated global model for use in the next communication round
		self.save_model(self.model, self.config.paths.model)

		# server weight pca
		server_weights = [self.flatten_weights(updated_weights)]
		server_weights = np.array(server_weights)
		server_weights_pca = self.pca.transform(server_weights)

		# update the weights of the selected devices and server to corresponding client id 
		# return next_state
		for i in range(len(sample_clients_ids)):
			self.pca_weights_clientserver[sample_clients_ids[i]] = clients_weights_pca[i]
		
		self.pca_weights_clientserver[-1] = server_weights_pca[0]

		next_state = self.pca_weights_clientserver.flatten()
		print("next_state.shape:", next_state.shape)
		next_state = next_state.tolist()
		
	
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

		# testing accuracy, updated pca_weights_clientserver
		return accuracy, next_state


	def dqn_reset_state(self):

		# randomly select k devices to conduct 1 round of FL to reset the states
		# only update the weights of the selected devices in self.pca_weights_clientserver_init

		# copy over again
		self.pca_weights_clientserver = self.pca_weights_clientserver_init.copy()

		# randomly select k devices, update the weights of the selected devices and server to get next_state
		accuracy, new_state = self.dqn_round(random=True) # updated self.pca_weights_clientserver
		self.prev_accuracy = accuracy

		return new_state

	def choose_action(self, state):
		
		# predict the q values for each action given the current state
		print("DQN choose action")
		q_values = self.dqn_model.predict([state], verbose=0)[0]

		#print("q_values:", q_values)
		# use a softmax function to convert the q values to probabilities
		probs = np.exp(q_values) / np.sum(np.exp(q_values))
		#print("probs:", probs)

		# add small value to each probability to avoid 0 probability
		#probs = probs + 0.000001

		# choose an action based on the probabilities
		action = np.random.choice(self.nA, p=probs)

		return action


	def train_episode(self, episode_ct, epsilon_current):

		# must reload the initial model for each episode
		self.load_model() # save initial global model

		# reset the state at beginning of each episode, randomly select k devices to reset the states
		state = self.dqn_reset_state() #++ reset the state at beginning of each episode, randomly select k devices to reset the states

		total_reward = 0
		com_rounds = 0
		final_acc = 0
		for t in range(self.max_steps):
			
			# action = self.epsilon_greedy(state, epsilon_current)
			action = self.choose_action(state)
			next_state, reward, done, acc = self.step(action) #++ during training, pick a client for next communication round
			print("episode_ct:", episode_ct, "step:", t, "acc:", acc, "action:", action, "reward:", reward, "done:", done)
			print()
			total_reward += reward
			com_rounds += 1
			final_acc = acc

			self.memorize(state, action, reward, next_state, done)
			self.replay() # sample a mini-batch from the replay buffer to train the DQN model
			state = next_state

			if done:
				break

			if t % self.target_update == 0:
				self.update_target_model()

		return total_reward, com_rounds, final_acc        


	def replay(self):
		
		if len(self.memory) > self.batch_size:
			print("Replaying...")
			sample_batch = random.sample(self.memory, self.batch_size)
			states = []
			target_q = []
			for state, action, reward, next_state, done in sample_batch:
				states.append(state)
				# need to use the model to predict the q values
				q = self.dqn_model.predict([state], verbose=0)[0]
				# print("rest")

				# then update the experiencd action value using the target model while keeping the other action values the same
				if done:
					q[action] = reward
				else:
					q[action] = reward + self.gamma * np.max(self.target_model.predict([next_state], verbose=0)[0])

				target_q.append(q)

			states = np.array(states)
			target_q = np.array(target_q)

			print("Fit dqn_model")
			self.dqn_model.fit(states, target_q, epochs=1, verbose=0)
			print("Replay done.")


	# Run multiple episodes of training for DQN
	def run(self):

		# initial profiling on all clients to get initial pca weights for each client and server model
		self.profile_all_clients(train_dqn=True)

		# write out the Episode, reward, round, accuracy 
		fn = self.config.dqn.rewards_log
		print("Reards logs written to:", fn)
		with open(fn, 'w') as f:
			f.write('Episode,Reward,Round,Accuracy\n')

		for i_episode in range(self.episode):

			print()
			t_start = time.time()
			# calculate the epsilon value for the current episode
			epsilon_current = self.config.dqn.epsilon_initial * pow(self.config.dqn.epsilon_decay, i_episode)
			epsilon_current = max(self.config.dqn.epsilon_min, epsilon_current)

			total_reward, com_round, final_acc = self.train_episode(i_episode+1, epsilon_current)

			t_end = time.time()
			print("Episode: {}/{}, total_reward: {}, com_round: {}, final_acc: {:.4f}, time: {:.2f} s".format(i_episode+1, self.episode, total_reward, com_round, final_acc, t_end - t_start))
			with open(fn, 'a') as f:
				f.write('{},{},{},{}\n'.format(i_episode, total_reward, com_round, final_acc))
			# save trained model to h5 file
			model_fn = self.config.dqn.saved_model + '_' +str(i_episode) + '.h5'
			self.dqn_model.save(model_fn)
			print("DQN model saved to:", model_fn)
		
		print("\nTraining finished!")



	# Federated learning phases
	def dqn_selection(self, action):

		sample_clients_list = [self.clients[action]]

		return sample_clients_list

	def calculate_reward(self, accuracy_this_round):
		
		target_accuracy = self.config.fl.target_accuracy
		xi = self.config.dqn.reward_xi # in article set to 64
		reward = xi**(accuracy_this_round - target_accuracy) -1

		return reward

	def calculate_reward_difference(self, cur_acc):
		
		prev_acc = self.prev_accuracy
		print("prev_acc:", prev_acc)
		print("cur_acc:", cur_acc)
		xi = self.config.dqn.reward_xi
		if cur_acc >= prev_acc:
			reward = xi**(cur_acc - prev_acc) # positive rewards based on improvement
		else:
			reward = - xi**(prev_acc - cur_acc) # negative rewards if testing acc drops

		return reward

	def step(self, action):

		accuracy, next_state = self.dqn_round(random=False, action=action) 
		
		# calculate the reward based on the accuracy and the number of communication rounds
		if self.config.dqn.reward_fun ==  "target":
			reward =self.calculate_reward(accuracy)
		elif self.config.dqn.reward_fun ==  "difference":
			reward =self.calculate_reward_difference(accuracy)
		
		# rest the prev_accuracy
		self.prev_accuracy = accuracy

		# determine if the episode is done based on if reaching the target testing accuracy        
		if accuracy >= self.config.fl.target_accuracy:
			done = True
		else:
			done = False

		return next_state, reward, done, accuracy


	def profiling(self, clients, train_dqn=False):

		# Configure clients to train on local data
		self.configuration(clients)
		sample_clients_ids = [client.client_id for client in clients]   
		# Train on local data for profiling purposes
		# Wait for reports
		while not all(super().get_clientlist().get_report_state(sample_clients_ids)):
			pass

		# Recieve client reports
		reports = self.reporting(clients)

		# Extract weights from reports
		# reduced_weights = [self.getPCAWeight(report.weights) for report in reports]
		clients_weights = [self.flatten_weights(report.weights) for report in reports] # list of numpy arrays
		clients_weights = np.array(clients_weights) # convert to numpy array

		clients_prefs = [report.pref for report in reports] # dominant class in each client
		print("clients_prefs:", clients_prefs)

		# print("clients_weights: ", clients_weights)
		# print("type of clients_weights[0]: ", type(clients_weights[0]))
		print("shape of clients_weights: ", clients_weights.shape)

		if train_dqn: # first time to initialize the PCA model during training of DQN
			# build the PCA transformer
			t_start = time.time()
			print("Start building the PCA transformer...")
			self.pca = PCA(n_components=self.pca_n_components)
			#self.pca = PCA(n_components=2)
			clients_weights_pca = self.pca.fit_transform(clients_weights)

			# dump clients_weights_pca out to pkl file for plotting
			clients_weights_pca_fn = 'output/clients_weights_pca.pkl'
			pk.dump(clients_weights_pca, open(clients_weights_pca_fn,"wb"))
			print("clients_weights_pca dumped to", clients_weights_pca_fn)    

			# dump clients_prefs
			clients_prefs_fn = 'output/clients_prefs.pkl'
			pk.dump(clients_prefs, open(clients_prefs_fn,"wb"))
			print("clients_prefs dumped to", clients_prefs_fn)        

			t_end = time.time()
			print("Built PCA transformer, time: {:.2f} s".format(t_end - t_start))

			# save pca model out to pickl file
			pca_model_fn = self.config.dqn.pca_model
			pk.dump(self.pca, open(pca_model_fn,"wb"))
			print("PCA model dumped to", pca_model_fn)

			#stop
		
		else: # during inference of DQN
			# directly use the pca model to transform the weights
			clients_weights_pca = self.pca.transform(clients_weights)

		# print("clients_weights_pca: ", clients_weights_pca)
		# print("type of clients_weights_pca[0]: ", type(clients_weights_pca[0]))
		print("shape of clients_weights_pca: ", clients_weights_pca.shape)

		# get server model updated weights based on reports from clients
		server_weights = [self.flatten_weights(self.aggregation(reports))]
		server_weights = np.array(server_weights)
		server_weights_pca = self.pca.transform(server_weights)

		# print("server_weights: ", server_weights)
		# print("server_weights_pca: ", server_weights_pca)
		print("shape of server_weights_pca: ", server_weights_pca.shape)

		return clients_weights_pca, server_weights_pca


	"""
	def getPCAWeight(self,weight):
		weight_flatten_array = self.flatten_weights(weight)
	   ## demision = int(math.sqrt(weight_flatten_array.size))
		# weight_flatten_array = np.abs(weight_flatten_array)
		# sorted_array = np.sort(weight_flatten_array)
		# reverse_array = sorted_array[::-1]

		demision = weight_flatten_array.size
		weight_flatten_matrix = np.reshape(weight_flatten_array,(10,int(demision/10)))
		
		pca = PCA(n_components=10)
		pca.fit_transform(weight_flatten_matrix)
		newWeight = pca.transform(weight_flatten_matrix)
		# newWeight = reverse_array[0:100]

		return  newWeight
	"""

	# Server operations
	def profile_all_clients(self, train_dqn):

		# all clients send updated weights to server, the server will do FedAvg
		# And then run  PCA and store the transformed weights

		print("Start profiling all clients...")

		assert len(self.clients)== self.config.clients.total

		# Perform profiling on all clients
		clients_weights_pca, server_weights_pca = self.profiling(self.clients, train_dqn)

		# save the initial pca weights for each client + server 
		self.pca_weights_clientserver_init = np.vstack((clients_weights_pca, server_weights_pca))
		print("shape of self.pca_weights_clientserver_init: ", self.pca_weights_clientserver_init.shape)
   
		# save a copy for later update in DQN training episodes
		self.pca_weights_clientserver = self.pca_weights_clientserver_init.copy() 

		print('self.pca_weights_clientserver.shape:', self.pca_weights_clientserver.shape)


class DQNServer(DQNTrainServer):
	"""
	FL server using pre-trained D-DQN agent to select top k devices based on the DQN's output
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