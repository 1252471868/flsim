import logging
import torch
import torch.nn as nn
import torch.optim as optim
import socket
import json
import sys
from threading import Lock
import numpy as np
import jsonpickle
import pickle
import dill
import random
from signal import signal, SIGPIPE, SIG_DFL
from TCPUDP_socket import TensorBuffer, Unflatten_Tensor
import utils.dists as dists  # pylint: disable=no-name-in-module
import struct
import TCPUDP_socket
from datetime import datetime


mutex = Lock()

class Client(object):
	"""Simulated federated learning client."""

	def __init__(self, client_id):
		self.client_id = client_id
		self.pref = -1

	def __repr__(self):
		return 'Client #{}: {} samples in labels: {}'.format(
			self.client_id, len(self.data), set([label for _, label in self.data]))
	# Set up client
	def boot(self, config):
		# logging.info('Booting {} server...'.format(self.config.server))
		self.seed(1234+self.client_id)
		self.set_config(config)
		model_path = self.config.paths.model
		# Add fl_model to import path
		sys.path.append(model_path)
		import fl_model
		self.model = fl_model.Net()
		self.connect_server()

	def seed(self, seed=1234):
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		np.random.seed(seed)
		random.seed(seed)
		torch.backends.cudnn.deterministic = True
# TCP/IP connect to server
	def connect_server(self):
		server_ip = self.config.server.socket.get('ip')
		# server_port = self.config.server.socket.get('port')
		# self.server_addr = (server_ip,server_port)
		# signal(SIGPIPE, SIG_DFL)
		# self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		# logging.info('Connecting to server')
		# self.client_socket.connect(self.server_addr)
		self.socket = TCPUDP_socket.TCPUDPClient(SERVER= server_ip, CLIENT_ID=self.client_id)
		logging.info('Server connected')
		# self.socket.send('INFO', msg_udp=[0])
		# for i in range(5):
		self.socket.send('ID', msg_tcp=self.client_id, msg_udp=[self.client_id])
		# if self.config.server.socket.get('protocol') == 'udp':
		self.socket.send('IP', msg_tcp=(self.client_id, self.socket.udp_addr))
		
		# self.send_data(self.client_socket, 'ID', self.client_id)
		

	# Set non-IID data configurations
	def set_bias(self, pref, bias):
		self.pref = pref
		self.bias = bias

	def set_shard(self, shard):
		self.shard = shard
	def set_socket(self, client_socket):
		self.client_socket = client_socket

	def set_config(self, config):
		self.config = config
	# Federated learning phases
	def set_data(self, data):
		# Extract from config
		do_test = self.do_test = self.config.clients.do_test
		test_partition = self.test_partition = self.config.clients.test_partition
		# Download data
		self.data = data
		# cmd, self.data = self.recv_data()
		# Extract trainset, testset (if applicable)
		if do_test:  # Partition for testset if applicable
			self.trainset = data[:int(len(data) * (1 - test_partition))]
			self.testset = data[int(len(data) * (1 - test_partition)):]
		else:
			self.trainset = data

	def set_testset(self, testset):
		self.do_test = True
		self.testset = testset

	def configure(self, raw_weights):
		import fl_model  # pylint: disable=import-error

		# Extract from config
		model_path = self.model_path = self.config.paths.model

		# Download from server

		# Extract machine learning task from config
		self.task = self.config.fl.task
		self.epochs = self.config.fl.epochs
		self.batch_size = self.config.fl.batch_size

		# Download most recent global model
		# path = model_path + '/global'
		baseline_weights = fl_model.extract_weights_noname(self.model)
		weights_flattened = TensorBuffer(baseline_weights)
		if self.config.server.socket.get('protocol') == 'udp':
			raw_weights = np.concatenate(raw_weights)
			indices = raw_weights[:, 0]
			weights = raw_weights[:, 1]
			weights_flattened.buffer[indices] = torch.from_numpy(weights)
		elif self.config.server.socket.get('protocol') == 'tcp':
			weights = raw_weights
			weights_flattened.buffer[:] = torch.from_numpy(weights)
		weights_unflattened = Unflatten_Tensor(weights_flattened, self.model)
		fl_model.load_weights_noname(self.model, weights_unflattened)
		# self.model = model
		# self.model=torch.load(model)
		# self.model.load_state_dict(torch.load(model))
		self.model.eval()

		# Create optimizer
		self.optimizer = fl_model.get_optimizer(self.model)

	def run(self):
		while True:
			# cmd, data = self.recv_data()
			cmd, data = self.socket.receive()
			if cmd == 'BIAS':
				pref = data[0]
				bias = data[1]
				self.set_bias(pref, bias)
				logging.info('Received bias')
			if cmd == 'CONFIG':
				self.set_config(data)
				logging.info('Received configuration')
			elif cmd == 'DATA':
				self.set_data(data)
				logging.info('Received data')
			elif cmd == 'TESTSET':
				self.set_testset(data)
				logging.info('Received testset')
			elif cmd == 'MODEL':
				logging.info('Received model')

				self.configure(data)
				{
					"train": self.train()
				}[self.task]
			elif cmd == 'PROBING':
				logging.info('probing training')
				self.configure(data)
				{
					"train": self.train(probing_training=True)
				}[self.task]

			elif cmd == 'END':
				logging.info('Completed.')
				# self.client_socket.close()
				break
			elif cmd == 'INFO':
				logging.info('{}'.format(data))
			
				
		# Perform federated learning task
		# {
		# 	"train": self.train()
		# }[self.task]

	# def get_report(self):
	#     # Report results to server.
	#     return self.upload(self.report)

	# Machine learning tasks
	def train(self, probing_training=False):
		import fl_model  # pylint: disable=import-error

		logging.info('Training on client #{}'.format(self.client_id))
		start_time = datetime.now()
		# Perform model training
		trainloader = fl_model.get_trainloader(self.trainset, self.batch_size)
		loss = fl_model.train(self.model, trainloader,
					   self.optimizer, self.epochs)
		time_diff = datetime.now() - start_time
		
		logging.info('Client {} completed'.format(self.client_id))
		# Extract model weights and biases
		weights =fl_model.extract_weights_noname(self.model)
		weights_flat = TensorBuffer(weights)
		
		# Generate report for server
		self.report = TCPUDP_socket.Report(self.client_id, len(self.data))
		# self.report.weights = weights
		# self.report.pref = int(self.pref.split(' - ')[0])
		self.report.pref = self.pref
		self.report.loss = loss
		self.report.training_latency = time_diff.total_seconds()
		self.report.comm_latency = datetime.now()
		#Since the model sizes are the same, the normalized communication cost is 1
		self.report.comm_cost = 1
		self.report.packet_num = np.ceil(len(weights_flat.buffer.numpy())/100)
		
		# Perform model testing if applicable
		if self.do_test:
			testloader = fl_model.get_testloader(self.testset, 1000)
			self.report.accuracy = fl_model.test(self.model, testloader)
		if probing_training==False:
			if self.config.server.socket.get('protocol') == 'tcp':
				self.socket.send('REPORT', msg_tcp=(self.report, weights_flat.buffer.numpy()))
			elif self.config.server.socket.get('protocol') == 'udp':
				self.socket.send('REPORT', msg_tcp=self.report, msg_udp=weights_flat.buffer.numpy())
		else:
			self.socket.send('PROBING_REPORT', msg_tcp=self.report)
		# self.send_data(self.client_socket, 'REPORT', self.report)
		logging.info('Client {} sends report'.format(self.client_id))

	def test(self):
		# Perform model testing
		raise NotImplementedError
	
	# def send_data(self, server, cmd, data):
	# 	msg = {}
	# 	msg['CMD'] = cmd
	# 	msg['DATA'] = data
	# 	# Convert data to JSON string
	# 	msg_string = dill.dumps(msg)
	# 	# msg_json = jsonpickle.encode(msg).encode()
	# 	# data_len = str(len(msg_json)).encode
	# 	data2send = struct.pack('>I', len(msg_string)) + msg_string
	# 	logging.info('CMD: {}'.format(cmd))
	# 	logging.info('Before dumping: {}'.format(sys.getsizeof(msg)))
	# 	logging.info('After dumping: {}'.format(sys.getsizeof(data2send)))
	# 	# server.sendall(data_len)
	# 	server.sendall(data2send)

	# def recv_data(self):
	# 	# datalength : 4 bytes
	# 	raw_data = self.recvall(4)
	# 	if not raw_data:
	# 		return None, None
	# 	msg_len = struct.unpack('>I', raw_data)[0]
	# 	msg_string = self.recvall(msg_len)
	# 	# logging.info('recv msgsize: {}'.format(sys.getsizeof(msg_json)))
	# 	msg = dill.loads(msg_string)
	# 	cmd = msg['CMD']
	# 	data = msg['DATA']
	# 	return cmd, data

	# #receive all raw data
	# def recvall(self, data_len):
	# 	data = bytearray()
	# 	while len(data)<data_len:
	# 		packet = self.client_socket.recv(data_len - len(data))
	# 		if not packet:
	# 			return None
	# 		data.extend(packet)
	# 	return data




