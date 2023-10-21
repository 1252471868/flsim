import logging
import torch
import torch.nn as nn
import torch.optim as optim
import socket
import json
import sys
from threading import Lock
import jsonpickle
import pickle
import dill
import random
from signal import signal, SIGPIPE, SIG_DFL
import utils.dists as dists  # pylint: disable=no-name-in-module
import struct

mutex = Lock()

class Client(object):
	"""Simulated federated learning client."""

	def __init__(self, client_id):
		self.client_id = client_id
		
	def __repr__(self):
		return 'Client #{}: {} samples in labels: {}'.format(
			self.client_id, len(self.data), set([label for _, label in self.data]))
	# Set up client
	def boot(self, config):
		# logging.info('Booting {} server...'.format(self.config.server))
		self.set_config(config)
		model_path = self.config.paths.model
		# Add fl_model to import path
		sys.path.append(model_path)
		self.connect_server()

# TCP/IP connect to server
	def connect_server(self):
		server_ip = self.config.server.socket.get('ip')
		server_port = self.config.server.socket.get('port')
		self.server_addr = (server_ip,server_port)
		signal(SIGPIPE, SIG_DFL)
		self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		logging.info('Connecting to server')
		self.client_socket.connect(self.server_addr)
		logging.info('Server connected')
		self.send_data(self.client_socket, 'ID', self.client_id)
		

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

	def configure(self, model):
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
		self.model = model
		# self.model.load_state_dict(torch.load(path))
		self.model.eval()

		# Create optimizer
		self.optimizer = fl_model.get_optimizer(self.model)

	def run(self):
		while True:
			cmd, data = self.recv_data()
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
			elif cmd == 'INFO':
				logging.info('{}'.format(data))
				
		# Perform federated learning task
		{
			"train": self.train()
		}[self.task]

	# def get_report(self):
	#     # Report results to server.
	#     return self.upload(self.report)

	# Machine learning tasks
	def train(self):
		import fl_model  # pylint: disable=import-error

		logging.info('Training on client #{}'.format(self.client_id))

		# Perform model training
		trainloader = fl_model.get_trainloader(self.trainset, self.batch_size)
		fl_model.train(self.model, trainloader,
					   self.optimizer, self.epochs)
		logging.info('Client {} completed'.format(self.client_id))
		# Extract model weights and biases
		weights = fl_model.extract_weights(self.model)
		
		# Generate report for server
		self.report = Report(self.client_id, len(self.data))
		self.report.weights = weights
		# Perform model testing if applicable
		if self.do_test:
			testloader = fl_model.get_testloader(self.testset, 1000)
			self.report.accuracy = fl_model.test(self.model, testloader)
		self.send_data(self.client_socket, 'REPORT', self.report)
		logging.info('Client {} sends report'.format(self.client_id))

	def test(self):
		# Perform model testing
		raise NotImplementedError
	
	def send_data(self, server, cmd, data):
		msg = {}
		msg['CMD'] = cmd
		msg['DATA'] = data
		# Convert data to JSON string
		msg_string = dill.dumps(msg)
		# msg_json = jsonpickle.encode(msg).encode()
		# data_len = str(len(msg_json)).encode
		data2send = struct.pack('>I', len(msg_string)) + msg_string
		# logging.info('send datasize: {}'.format(sys.getsizeof(msg_json)))
		# server.sendall(data_len)
		server.sendall(data2send)

	def recv_data(self):
		# datalength : 4 bytes
		raw_data = self.recvall(4)
		if not raw_data:
			return None, None
		msg_len = struct.unpack('>I', raw_data)[0]
		# logging.info('data len json: {} size:{}'.format(data_len_json,len(data_len_json)))
		# data_len = json.loads(data_len_json)
		# data_len = jsonpickle.decode(data_len_json)
		# logging.info('recv datasize: {}'.format(msg_len))
		# data is split across multiple recv()
		msg_string = self.recvall(msg_len)
		# logging.info('recv msgsize: {}'.format(sys.getsizeof(msg_json)))
		msg = dill.loads(msg_string)
		cmd = msg['CMD']
		data = msg['DATA']
		return cmd, data

	#receive all raw data
	def recvall(self, data_len):
		data = bytearray()
		while len(data)<data_len:
			packet = self.client_socket.recv(data_len - len(data))
			if not packet:
				return None
			data.extend(packet)
		return data




class Report(object):
	"""Federated learning client report."""

	def __init__(self, id = 0, num_sample = 0):
		self.weights = []
		self.client_id = id
		self.num_samples = num_sample
	
