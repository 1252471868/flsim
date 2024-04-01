import client
import load_data
import logging
import numpy as np
import pickle
import random
import sys
from threading import Thread, Lock
import torch
from TCPUDP_socket import TensorBuffer
import utils.dists as dists  # pylint: disable=no-name-in-module
import socket
import socketserver
import io
import json
import jsonpickle
import dill
from signal import signal, SIGPIPE, SIG_DFL
import struct
import TCPUDP_socket
from datetime import datetime

mutex = Lock()
# clients list for server

	



class Server(object):
	"""Basic federated learning server."""

	def __init__(self, config, case_name):
		self.config = config
		self.case_name = case_name

	# Set up server
	def boot(self):
		logging.info('Booting {} server...'.format(self.config.server))
		self.seed()
		model_path = self.config.paths.model
		client_cfg = self.config.clients

		# Add fl_model to import path
		sys.path.append(model_path)

		# Set up server
		self.load_data()
		self.load_model()
		server_ip = self.config.server.socket.get('ip')
		server_port = self.config.server.socket.get('port')
		self.socket = TCPUDP_socket.TCPUDPServer(SERVER=server_ip,NUM_CLIENTS=client_cfg.total)
		self.connect_clients(client_cfg)

	def seed(self, seed=1234):
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		np.random.seed(seed)
		random.seed(seed)
		torch.backends.cudnn.deterministic = True


	def load_data(self):
		import fl_model  # pylint: disable=import-error

		# Extract config for loaders
		config = self.config

		# Set up data generator
		generator = fl_model.Generator()
		# Generate data
		data_path = self.config.paths.data
		data = generator.generate(data_path)
		labels = generator.labels

		logging.info('Dataset size: {}'.format(
			sum([len(x) for x in [data[label] for label in labels]])))
		logging.info('Labels ({}): {}'.format(
			len(labels), labels))

		# Set up data loader
		self.loader = {
			'basic': load_data.Loader(config, generator),
			'bias': load_data.BiasLoader(config, generator),
			'shard': load_data.ShardLoader(config, generator)
		}[self.config.loader]

		logging.info('Loader: {}, IID: {}'.format(
			self.config.loader, self.config.data.IID))


	def load_model(self):
		import fl_model  # pylint: disable=import-error

		model_path = self.config.paths.model
		model_type = self.config.model

		logging.info('Model: {}'.format(model_type))

		# Set up global model
		self.model = fl_model.Net()
		self.save_model(self.model, model_path)

		# Extract flattened weights (if applicable)
		if self.config.paths.reports:
			self.saved_reports = {}
			self.save_reports(0, [])  # Save initial model

	# def msg_handler(self, server_ip,server_port):
	# 	logging.info('Server started, waiting for connecting')
	# 	# if self.config.server.socket.get('protocol') == 'tcp':
	# 	socketserver.TCPServer.allow_reuse_address = True	
	# 	s=socketserver.ThreadingTCPServer((server_ip,server_port), TCPServerHandler)
	# 	# elif self.config.server.socket.get('protocol') == 'udp':
	# 	# 	socketserver.UDPServer.allow_reuse_address = True
	# 	# 	s=socketserver.ThreadingUDPServer((server_ip,server_port), TCPServerHandler)
	# 	s.serve_forever()

	def connect_clients(self, client_cfg):
		IID = self.config.data.IID
		labels = self.loader.labels
		loader = self.config.loader
		loading = self.config.data.loading
		self.socket.start(client_cfg.total)
		# self.socket.listen_K_clients(client_cfg.total)
		# signal(SIGPIPE, SIG_DFL)
		# server_ip = self.config.server.socket.get('ip')
		# server_port = self.config.server.socket.get('port')
		# msg_thread = Thread(target=self.msg_handler, args=(server_ip,server_port,))
		# msg_thread.start()
		#Wait for connection
		while len(self.socket.clients_list.id_list) < client_cfg.total :
			pass
 # Make simulated clients
		if not IID:  # Create distribution for label preferences if non-IID
			dist = {
				"uniform": dists.uniform(client_cfg.total, len(labels)),
				"normal": dists.normal(client_cfg.total, len(labels))
			}[client_cfg.label_distribution]
			random.shuffle(dist)  # Shuffle distribution
		clients = []
		for client_id in range(client_cfg.total):
			# Create new client
			new_client = client.Client(client_id)

			if not IID:  # Configure clients for non-IID data
				if self.config.data.bias:
					# Bias data partitions
					bias = self.config.data.bias
					# Choose weighted random preference
					pref = random.choices(labels, dist)[0]

					# Assign preference, bias config
					self.set_client_bias(new_client, pref, bias)
					# new_client.set_bias(pref, bias)
				elif self.config.data.shard:
					# Shard data partitions
					shard = self.config.data.shard
					# Assign shard config
					new_client.set_shard(shard)
			clients.append(new_client)

		logging.info('Total clients: {}'.format(len(clients)))

		if loader == 'bias':
			logging.info('Label distribution: {}'.format(
				[[client.pref for client in clients].count(label) for label in labels]))



		# self.send_data(0,'INFO','test')
		# self.send_data(1,'INFO','test')
		if loading == 'static':
			if loader == 'shard':  # Create data shards
				self.loader.create_shards()
			# Send data partition to all clients
			[self.set_client_data(client) for client in clients]
		self.clients = clients


	# Run federated learning
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

		# Perform rounds of federated learning
		for round in range(1, rounds + 1):
			logging.info('**** Round {}/{} ****'.format(round, rounds))

			if self.config.fl.probing_train:
				# logging.info('Probing training.....\n')
				self.probing_training()
			# Run the federated learning round
			accuracy = self.round()
			
			with open('output/'+self.case_name+'.csv', 'a') as f:
				f.write('{},{:.4f}'.format(round, accuracy*100)+'\n')

			# Break loop when target accuracy is met
			if target_accuracy and (accuracy >= target_accuracy):
				logging.info('Target accuracy reached.')
				break

		if reports_path:
			with open(reports_path, 'wb') as f:
				pickle.dump(self.saved_reports, f)
			logging.info('Saved reports: {}'.format(reports_path))
		
		for client in self.clients:
			# self.send_data(client.client_id, 'END', 0)
			self.socket.send(client.client_id, 'END', msg_tcp=0)
		logging.info('*******************Completed*********************')

	def round(self):
		import fl_model  # pylint: disable=import-error

		# Select clients to participate in the round
		sample_clients = self.selection()
		sample_clients_id = [client.client_id for client in sample_clients]
		logging.info("select clients: {}".format(sample_clients_id))
		# Configure sample clients
		self.configuration(sample_clients)

		# Wait for reports
		# self.socket.listen_K_clients(len(sample_clients))
		while not all(self.socket.clients_list.get_report_state(sample_clients_id)):
			pass
		self.socket.clients_list.clear_report_state()

		# Recieve client updates
		reports = self.reporting(sample_clients)
		comm_latencies = np.zeros(self.config.clients.total)
		rest_training_latencies = np.zeros(self.config.clients.total)
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
		return accuracy

	# Federated learning phases

	def probing_training(self):
		import fl_model  # pylint: disable=import-error
		clients = self.clients
		# Configure clients to train on local data
		logging.info('Probing training...')
		self.configuration(clients, probing_training=True)     
		sample_clients_ids = [client.client_id for client in clients]   
		# Train on local data for profiling purposes
		# Wait for reports
		while not all(self.socket.clients_list.get_report_state(sample_clients_ids)):
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


	def selection(self):
		# Select devices to participate in round
		clients_per_round = self.config.clients.per_round

		# Select clients randomly
		sample_clients = [client for client in random.sample(
			self.clients, clients_per_round)]

		return sample_clients

	def configuration(self, sample_clients, probing_training = False):
		import fl_model
		loader_type = self.config.loader
		loading = self.config.data.loading

		if loading == 'dynamic':
			# Create shards if applicable
			if loader_type == 'shard':
				self.loader.create_shards()

		# Configure selected clients for federated learning task
		weights = fl_model.extract_weights_noname(self.model)
		weights_flat = TensorBuffer(weights)
		for client in sample_clients:
			if loading == 'dynamic':
				self.set_client_data(client)  # Send data partition to client

			# Extract config for client
			# config = self.config

			# Continue configuraion on client
			# client.configure(config)
			# buffer = io.BytesIO()
			# torch.save(self.model.state_dict(), buffer)
			# model2send = buffer.getvalue()
			if probing_training == False:
				if self.config.server.socket.get('protocol') == 'tcp':
					# Convert tensor into narray, otherwise the size of the encoded tensor will be huge
					self.socket.send(client.client_id, 'MODEL', msg_tcp=weights_flat.buffer.numpy())
				elif self.config.server.socket.get('protocol') == 'udp':
					self.socket.send(client.client_id, 'MODEL', msg_udp=weights_flat.buffer.numpy())
			else:
				if self.config.server.socket.get('protocol') == 'tcp':
					# Convert tensor into narray, otherwise the size of the encoded tensor will be huge
					self.socket.send(client.client_id, 'PROBING', msg_tcp=weights_flat.buffer.numpy())
				elif self.config.server.socket.get('protocol') == 'udp':
					self.socket.send(client.client_id, 'PROBING', msg_udp=weights_flat.buffer.numpy())
			# self.send_data(client.client_id, 'MODEL', self.model)

	def reporting(self, sample_clients):
		# Recieve reports from sample clients
		reports = [self.socket.clients_list.get_report(client.client_id) for client in sample_clients]
		logging.info('Reports recieved: {}'.format(len(reports)))
		assert len(reports) == len(sample_clients)

		return reports

	def aggregation(self, reports):
		return self.federated_averaging(reports)

	# Report aggregation
	def extract_client_updates(self, report):
		import fl_model  # pylint: disable=import-error

		# Extract baseline model weights
		# baseline_weights = fl_model.extract_weights(self.model)
		baseline_weights = fl_model.extract_weights_noname(self.model)
		baseline_flattened = TensorBuffer(baseline_weights) 
		weights_flattened = TensorBuffer(baseline_weights)
		# Extract weights from reports
		if self.config.server.socket.get('protocol') == 'udp':
			indices = report.weights[:, 0]
			weights = report.weights[:, 1]
			weights_flattened.buffer[indices] = torch.from_numpy(weights)
		elif self.config.server.socket.get('protocol') == 'tcp':
			weights = report.weights
			weights_flattened.buffer[:] = torch.from_numpy(weights)
			
		# weights = [report.weights for report in reports]

		# Calculate updates from weights
		updates = []
		for i in range(len(weights_flattened)):
			delta = weights_flattened[i] - baseline_flattened[i]
			updates.append(delta)
		# for weight in weights:
		# 	update = []
		# 	for i, (name, weight) in enumerate(weight):
		# 		bl_name, baseline = baseline_weights[i]

		# 		# Calculate update
		# 		delta = weight - baseline
		# 		update.append((name, delta))
		# 	updates.append(update)

		return updates

	def federated_averaging(self, reports):
		import fl_model  # pylint: disable=import-error

		# Extract updates from reports
		updates = [self.extract_client_updates(report) for report in reports ]
		# updates = self.extract_client_updates(reports)

		# Extract total number of samples
		total_samples = sum([report.num_samples for report in reports])

		# Perform weighted averaging
		avg_update = [torch.zeros(x.size())  # pylint: disable=no-member
					  for x in updates[0]]
		for i, update in enumerate(updates):
			num_samples = reports[i].num_samples
			for j, delta in enumerate(update):	
				avg_update[j] += delta * (num_samples / total_samples)
			# for j, (_, delta) in enumerate(update):
			# 	# Use weighted average by number of samples
			# 	avg_update[j] += delta * (num_samples / total_samples)

		# Extract baseline model weights
		baseline_weights = fl_model.extract_weights(self.model)

		# Load updated weights into model
		updated_weights = []
		for i, (name, weight) in enumerate(baseline_weights):
			updated_weights.append((name, weight + avg_update[i]))

		return updated_weights

	def accuracy_averaging(self, reports):
		# Get total number of samples
		total_samples = sum([report.num_samples for report in reports])

		# Perform weighted averaging
		accuracy = 0
		for report in reports:
			accuracy += report.accuracy * (report.num_samples / total_samples)

		return accuracy

	# Server operations
	@staticmethod
	def flatten_weights(weights):
		# Flatten weights into vectors
		weight_vecs = []
		for _, weight in weights:
			weight_vecs.extend(weight.flatten().tolist())

		return np.array(weight_vecs)

	def set_client_data(self, client):
		loader = self.config.loader
		# Get data partition size
		if loader != 'shard':
			if self.config.data.partition.get('size'):
				partition_size = self.config.data.partition.get('size')
			elif self.config.data.partition.get('range'):
				start, stop = self.config.data.partition.get('range')
				partition_size = random.randint(start, stop)
		# Extract data partition for client
		if loader == 'basic':
			data = self.loader.get_partition(partition_size)
		elif loader == 'bias':
			data = self.loader.get_partition(partition_size, client.pref)
		elif loader == 'shard':
			data = self.loader.get_partition()
		else:
			logging.critical('Unknown data loader type')
		# Send data to client
		self.socket.send(client.client_id, 'CONFIG', msg_tcp=self.config)
		self.socket.send(client.client_id, 'DATA', msg_tcp=data)
		# self.send_data(client.client_id, 'CONFIG', self.config)
		# self.send_data(client.client_id, 'DATA', data)
		# client.set_data(data, self.config)

	def set_client_bias(self, client, pref, bias):
		client.set_bias(pref, bias)
		self.socket.send(client.client_id, 'BIAS', msg_tcp=(pref, bias))
		# self.send_data(client.client_id, 'BIAS', (pref, bias))

	def set_client_config(self, client):		
		self.socket.send(client.client_id, 'CONFIG', msg_tcp=self.config)
		# self.send_data(client.client_id, 'CONFIG', self.config)

	def save_model(self, model, path):
		path += '/global'
		torch.save(model.state_dict(), path)
		logging.info('Saved global model: {}'.format(path))

	def save_reports(self, round, reports):
		import fl_model  # pylint: disable=import-error

		if reports:
			self.saved_reports['round{}'.format(round)] = [(report.client_id, self.flatten_weights(
				report.weights)) for report in reports]

		# Extract global weights
		self.saved_reports['w{}'.format(round)] = self.flatten_weights(
			fl_model.extract_weights(self.model))

	# def send_data(self, client_id, cmd, data):
	# 	msg = {}
	# 	msg['CMD'] = cmd
	# 	msg['DATA'] = data
	# 	# Convert data to JSON string
	# 	# if cmd == 'MODEL':
	# 	# 	data2send = data
	# 	# 	data_string = dill.dumps(data)
	# 	# 	logging.info('Model Before dumping: {}'.format(sys.getsizeof(data)))
	# 	# 	logging.info('Model After dumping: {}'.format(sys.getsizeof(data_string)))
	# 	# 	client_socket = clients_list.get_socket(client_id)
	# 	# 	# server.sendall(data_len)
	# 	# 	client_socket.sendall(data2send)
	# 	# else:
	# 	msg_string = dill.dumps(msg)
	# 	# msg_json = jsonpickle.encode(msg).encode()
	# 	# data_len = str(len(msg_json)).encode
	# 	data2send = struct.pack('>I', len(msg_string)) + msg_string
	# 	logging.info('CMD: {}'.format(cmd))
	# 	logging.info('Before dumping: {}'.format(sys.getsizeof(msg)))
	# 	logging.info('After dumping: {}'.format(sys.getsizeof(data2send)))
	# 	client_socket = clients_list.get_socket(client_id)
	# 	# server.sendall(data_len)
	# 	client_socket.sendall(data2send)
	
	def get_clientlist(self):
		return self.socket.clients_list

# class TCPServerHandler(socketserver.BaseRequestHandler):
# 	def handle(self):
# 		while True:
# 			try:
# 				cmd, data = self.recv_data()
# 				# New clients will send ID to the server
# 				if cmd == 'ID':
# 					client_id = data
# 					clients_list.add_client(client_id, self.request)
# 					logging.info('Client {} connected'.format(client_id))
# 					logging.info('Address: {}'.format(self.client_address))
# 				if cmd == 'REPORT':
# 					report = data
# 					client_id = clients_list.get_id(self.request)
# 					clients_list.set_report(client_id, report)
# 					clients_list.set_report_state(client_id)
# 					logging.info('Received report from client {} '.format(client_id))
# 				if cmd == 'INFO':
# 					logging.info('{}'.format(data))
# 			except Exception as e:
# 				print(e)
# 				print('exception occured ,go to accpet next ')
# 				break
			
# 	def recv_data(self):
# 		# datalength : 4 bytes
# 		raw_data = self.recvall(4)
# 		if not raw_data:
# 			return None, None
# 		msg_len = struct.unpack('>I', raw_data)[0]
# 		# logging.info('data len json: {} size:{}'.format(data_len_json,len(data_len_json)))
# 		# data_len = json.loads(data_len_json)
# 		# data_len = jsonpickle.decode(data_len_json)
# 		# logging.info('recv datasize: {}'.format(msg_len))
# 		# data is split across multiple recv()
# 		msg_string = self.recvall(msg_len)
# 		# logging.info('recv msgsize: {}'.format(sys.getsizeof(msg_json)))
# 		msg = dill.loads(msg_string)
# 		# msg =  jsonpickle.decode(msg_json.decode())
# 		cmd = msg['CMD']
# 		data = msg['DATA']
# 		return cmd, data
	
# 	#receive all raw data
# 	def recvall(self, data_len):
# 		data = bytearray()
# 		while len(data)<data_len:
# 			packet = self.request.recv(data_len - len(data))
# 			if not packet:
# 				return None
# 			data.extend(packet)
# 		return data

# 	def send_data(self, client_id, cmd, data):
# 		msg = {}
# 		msg['CMD'] = cmd
# 		msg['DATA'] = data
# 		# Convert data to JSON string
# 		# if cmd == 'MODEL':
# 		# 	data2send = data
# 		# 	data_string = dill.dumps(data)
# 		# 	logging.info('Model Before dumping: {}'.format(sys.getsizeof(data)))
# 		# 	logging.info('Model After dumping: {}'.format(sys.getsizeof(data_string)))
# 		# 	client_socket = clients_list.get_socket(client_id)
# 		# 	# server.sendall(data_len)
# 		# 	client_socket.sendall(data2send)
# 		# else:
# 		msg_string = dill.dumps(msg)
# 		# msg_json = jsonpickle.encode(msg).encode()
# 		# data_len = str(len(msg_json)).encode
# 		data2send = struct.pack('>I', len(msg_string)) + msg_string
# 		logging.info('CMD: {}'.format(cmd))
# 		logging.info('Before dumping: {}'.format(sys.getsizeof(msg)))
# 		logging.info('After dumping: {}'.format(sys.getsizeof(data2send)))
# 		client_socket = clients_list.get_socket(client_id)
# 		# server.sendall(data_len)
# 		client_socket.sendall(data2send)


# class UDPServerHandler(socketserver.BaseRequestHandler):
# 	def handle(self):
# 		cmd, data = self.recv_data()
# 		if cmd == 'ID':
# 			client_id = data
# 			clients_list.add_client(client_id, 0, self.client_address)
# 			print('Client {} connected'.format(client_id))
# 			print('Address: {}'.format(self.client_address))
# 		elif cmd == 'ERROR':
# 			print('error!')
# 		self.send_data(client_id, 'ACK', 'server received')

# 	def send_data(self, client_id, cmd, data):
# 		msg = {}
# 		msg['CMD'] = cmd
# 		msg['DATA'] = data
# 		# Convert data to JSON string
# 		msg_string = dill.dumps(msg)
# 		# msg_json = jsonpickle.encode(msg).encode()
# 		# data_len = str(len(msg_json)).encode
# 		data2send = msg_string
# 		# logging.info('send datasize: {}'.format(sys.getsizeof(msg_json)))
# 		client_addr = clients_list.get_addr(client_id)
# 		# server.sendall(data_len)
# 		self.request[1].sendto(data2send, client_addr)

# 	def recv_data(self):
# 		msg = dill.loads(self.request[0])
# 		cmd = msg['CMD']
# 		data = msg['DATA']
# 		return cmd, data



