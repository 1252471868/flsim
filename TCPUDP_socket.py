import io
import time
import torch
import socket
import threading
import logging
import dill
import struct
import numpy as np
from signal import signal, SIGPIPE, SIG_DFL

BUFFER = 1024 * 64

class ClientList(object):
	def __init__(self):
		self.id_list = []
		self.socket_list = []
		self.addr_list = []
		self.report_state_list = []
		self.report_list = []
	
	def add_client(self, client_id, client_socket, client_addr):
		self.id_list.append(client_id)
		self.socket_list.append(client_socket)		
		self.report_state_list.append(False)
		self.report_list.append(None)
		self.addr_list.append(client_addr)

	def get_addr(self, client_id):
		return self.addr_list[self.id_list.index(client_id)]

	def get_socket(self, client_id):
		return self.socket_list[self.id_list.index(client_id)]
	
	def get_id(self, client_socket):
		return self.id_list[self.socket_list.index(client_socket)]
	
	def set_report_state(self, client_id):
		self.report_state_list[self.id_list.index(client_id)] = True

	def clear_report_state(self):
		self.report_state_list = [False for r in self.report_state_list]

	def get_report_state(self, client_id):
		state = [self.report_state_list[self.id_list.index(id)] for id in client_id]
		return state
	
	def set_report(self, client_id, report):
		self.report_list[self.id_list.index(client_id)] = report

	def get_report(self, client_id):
		return self.report_list[self.id_list.index(client_id)]


class TensorBuffer:
	"""
	Class to flatten and deflatten the gradient vector.
	"""

	def __init__(self, tensors):
		indices = [0]
		for tensor in tensors:
			new_end = indices[-1] + tensor.nelement()
			indices.append(new_end)

		self._start_idx = indices[:-1]
		self._end_idx = indices[1:]
		self._len_tensors = len(tensors)
		self._tensor_shapes = [tensor.size() for tensor in tensors]

		self.buffer = torch.cat([tensor.view(-1) for tensor in tensors])

	def __getitem__(self, index):
		return self.buffer[self._start_idx[index] : self._end_idx[index]].view(self._tensor_shapes[index])

	def __len__(self):
		return self._len_tensors

def Unflatten_Tensor(flattened_tensors, model):
	buffer = []
	for idx, _ in enumerate(model.parameters()):
		buffer.append(flattened_tensors[idx])
	return buffer


class Report(object):
	"""Federated learning client report."""

	def __init__(self, id = 0, num_sample = 0):
		self.weights = []
		self.client_id = id
		self.num_samples = num_sample
	

class TCPUDPServer:
	def __init__(
		self,
		SERVER=socket.gethostbyname(socket.gethostname()),
		TCP_PORT=5050,
		UDP_PORT=5060,
		NUM_CLIENTS=1,
		TIMEOUT=2,
		GRADIENT_SIZE=14728266,
		CHUNK=100,
		DELAY=3e-3,
		SEED=42,
	):
		self.SERVER = SERVER
		self.TCP_PORT = TCP_PORT
		self.UDP_PORT = UDP_PORT

		self.TIMEOUT = TIMEOUT
		self.NUM_CLIENTS = NUM_CLIENTS
		self.GRADIENT_SIZE = GRADIENT_SIZE
		self.CHUNK = CHUNK
		self.DELAY = DELAY
		self.SEED = SEED

		self.TCP_ADDR = (SERVER, TCP_PORT)
		self.UDP_ADDR = (SERVER, UDP_PORT)

		self.START_OF_MESSAGE = -float("inf")
		self.END_OF_MESSAGE = float("inf")

		self.clients_list = ClientList()
		signal(SIGPIPE, SIG_DFL)
		self.serverTCP = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.serverTCP.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self.serverTCP.bind(self.TCP_ADDR)
		logging.info('Server: {}'.format(self.TCP_ADDR))

		self.UDP_PORT_LIST = [UDP_PORT + i for i in range(NUM_CLIENTS)]
		self.UDP_ADDR_LIST = [(SERVER, UDP_PORT_ELEMENT) for UDP_PORT_ELEMENT in self.UDP_PORT_LIST]
		self.serverUDP = [None] * NUM_CLIENTS

		for ind in range(NUM_CLIENTS):
			self.serverUDP[ind] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
			self.serverUDP[ind].bind(self.UDP_ADDR_LIST[ind])
			self.serverUDP[ind].settimeout(self.TIMEOUT)

		# self.accumulated_gradient = torch.zeros(GRADIENT_SIZE)

	def encode(self, data):
		encoded = dill.dumps(data)
		return encoded

	def decode(self, buffer):
		data = dill.loads(buffer)
		return data

	def sendTCP(self, conn, data):
		encoded_message = self.encode(data)
		data2send = struct.pack('>I', len(encoded_message)) + encoded_message
		conn.send(data2send)

		# time.sleep(self.DELAY)
		# self.send_EOT(conn)

		return

	def sendTCP_SOT(self, conn):
		self.sendTCP(self.START_OF_MESSAGE, conn)
		return
	

	def sendTCP_EOT(self, conn):
		self.sendTCP(self.END_OF_MESSAGE, conn)
		return
	
	def sendUDP(self, client_id, data):
		messages = np.array_split(data,  np.arange(self.CHUNK, len(data), self.CHUNK))
		# messages = data.split(self.CHUNK)
		addr = self.clients_list.get_addr(client_id)
		for ind, message in enumerate(messages):
			indexed_message = np.concatenate((np.array([ind * self.CHUNK], dtype=np.float32), message))
			encoded_message = self.encode(indexed_message)
			rnd = np.random.uniform(0,1)
			# if rnd<0.95:
			self.serverUDP[client_id].sendto(encoded_message, addr)
			time.sleep(self.DELAY)

		return

	def send(self, client_id, cmd, msg_udp=None, msg_tcp=None):
		client_socket = self.clients_list.get_socket(client_id)
		# self.sendTCP_SOT(client_socket)
		msg_type='tcp'
		if msg_udp is not None and msg_tcp is None:
			msg_type='udp'
		elif msg_udp is not None and msg_tcp is not None:
			msg_type='tcpudp'
			
		self.sendTCP(client_socket, (self.START_OF_MESSAGE,cmd, msg_type))

		if msg_type != 'udp':
			self.sendTCP(client_socket, msg_tcp)
		if msg_type != 'tcp':
			self.sendUDP(client_id, msg_udp)

		self.sendTCP(client_socket, (self.END_OF_MESSAGE,cmd, msg_type))
		return
	

	def recv_TCP(self, conn):
		raw_data = self.recvall(conn, 4)
		if not raw_data:
			return None
		msg_len = struct.unpack('>I', raw_data)[0]
		msg = self.recvall(conn, msg_len)
		decoded_msg = self.decode(msg)
		return decoded_msg
		
		#tcp receive all raw data
	def recvall(self, conn, data_len):
		data = bytearray()
		while len(data)<data_len:
			packet = conn.recv(data_len - len(data))
			if not packet:
				return None
			data.extend(packet)
		return data
	
	def recv_UDP(self, client_id):
		buffer = []
		addr = None
		while True:
			try:
				msg, addr = self.serverUDP[client_id].recvfrom(BUFFER)
				# logging.info('Client addr: {}'.format(addr))
				# self.send(client_id, 'INFO', msg_udp=[333])
			except socket.error:
				break

			# if client_id not in self.clients_list.id_list:
			# 	self.clients_list.add_client(client_id, None, addr)

			try:
				decoded_msg = self.decode(msg)
			except:
				continue

			start_index = decoded_msg[0]
			start_index_indices = np.arange(start_index, start_index + len(decoded_msg) - 1, dtype=np.float32)
			start_index_indices_grad = np.vstack([start_index_indices, decoded_msg[1:]]).T
			buffer.append(start_index_indices_grad)
		logging.info('UDP: received {} packets'.format(len(buffer)))
		return buffer, addr
 
	def receive(self, conn):
		msg = []
		buffer = []
		data = []
		udp_addr = None
		readnext = True
		while readnext:
			try:
				msg = self.recv_TCP(conn)
				if msg is None:
					continue
				flag = msg[0]
				cmd = msg[1]
				msg_type = msg[2]
				client_id = msg[3]
				# conn.setblocking(False)
			except socket.error:
				pass

			if np.isinf(flag) and np.sign(flag) > 0:
				# conn.setblocking(True)
				readnext = False

			if np.isinf(flag) and np.sign(flag) < 0:
				if msg_type == 'tcp':
					data = self.recv_TCP(conn)
				elif msg_type == 'udp':
					data, udp_addr = self.recv_UDP(client_id)
				elif msg_type == 'tcpudp':
					buffer = self.recv_TCP(conn)
					data.append(buffer)
					buffer, udp_addr = self.recv_UDP(client_id)
					data.append(buffer)
		return cmd, data, udp_addr
		# self.accumulated_gradient[indices] += gradient

		return

	def unpack_weights(self, raw_weights):
		raw_weights = np.concatenate(raw_weights)
		indices = raw_weights[:, 0]
		weights = raw_weights[:, 1]
		return 

	def msg_handler(self, conn, addr):
		# conn.setblocking(1)
		while True:
			try:
				cmd, data, udp_addr = self.receive(conn)
				# New clients will send ID to the server
				if cmd == 'ID':
					client_id = data[0]
					self.clients_list.add_client(client_id, conn, udp_addr)
					logging.info('Client {} connected'.format(client_id))
					logging.info('TCP Address: {}'.format(addr))
					logging.info('UDP Address: {}'.format(udp_addr))
				elif cmd == 'REPORT':
					report = data[0]
					# weights = data[1]
					if udp_addr is None:
						weights = data[1]
					else:
						weights = np.concatenate(data[1], dtype=np.float32)
					report.weights = weights
					client_id = self.clients_list.get_id(conn)
					self.clients_list.set_report(client_id, report)
					self.clients_list.set_report_state(client_id)
					logging.info('Received report from client {} '.format(client_id))
				else:
					logging.info('{}: {}'.format(cmd, data))

			except Exception as e:
				print(e)
				print('exception occured ,go to accpet next ')
				break
			
		

	def start(self, num_sample):
		self.serverTCP.listen()
		print(f"[LISTENING] Server is listening on {self.SERVER}")
		logging.info('Listening to {} clients'.format(num_sample))
		receiving_threads = []
		client_count = 0
		while client_count < num_sample:
			conn, addr = self.serverTCP.accept()
			client_count += 1

			receiving_thread = threading.Thread(target=self.msg_handler, args=(conn, addr))
			receiving_threads.append(receiving_thread)
			receiving_thread.start()

			# for thread in receiving_threads:
			# 	thread.join()
		return

	# def listen_K_clients(self, num_sample):
	# 	logging.info('Listening to {} clients'.format(num_sample))
	# 	try:
	# 		receiving_threads = []
	# 		client_count = 0

	# 		while client_count < num_sample:
	# 			conn, addr = self.serverTCP.accept()
	# 			client_count += 1

	# 			receiving_thread = threading.Thread(target=self.msg_handler, args=(conn, addr))
	# 			receiving_threads.append(receiving_thread)
	# 			receiving_thread.start()

	# 		for thread in receiving_threads:
	# 			thread.join()
	# 	except KeyboardInterrupt:
	# 		self.stop()

	def stop(self):
		self.serverTCP.shutdown(1)
		self.serverTCP.close()

		for ind in range(self.NUM_CLIENTS):
			self.serverUDP[ind].close()

		return


class TCPUDPClient:
	def __init__(
		self,
		SERVER=socket.gethostbyname(socket.gethostname()),
		TCP_PORT=5050,
		UDP_PORT=5060,
		TIMEOUT=2,
		GRADIENT_SIZE=14728266,
		CHUNK=100,
		DELAY=3e-3,
		SEED=42,
		CLIENT_ID=0,
	):
		self.SERVER = SERVER
		self.TCP_PORT = TCP_PORT
		self.UDP_PORT = UDP_PORT

		self.TIMEOUT = TIMEOUT
		self.GRADIENT_SIZE = GRADIENT_SIZE
		self.CHUNK = CHUNK
		self.DELAY = DELAY
		self.SEED = SEED
		self.CLIENT_ID = CLIENT_ID

		self.TCP_ADDR = (SERVER, TCP_PORT)
		self.UDP_ADDR = (SERVER, UDP_PORT + CLIENT_ID)

		self.START_OF_MESSAGE =-float("inf")
		self.END_OF_MESSAGE = float("inf")

		logging.info('Connecting to server {}'.format(self.TCP_ADDR))
		signal(SIGPIPE, SIG_DFL)
		self.clientTCP = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.clientTCP.connect(self.TCP_ADDR)
		# self.clientTCP.setblocking(1)

		self.clientUDP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		self.clientUDP.settimeout(self.TIMEOUT)

	def encode(self, data):
		encoded = dill.dumps(data)
		return encoded

	def decode(self, buffer):
		data = dill.loads(buffer)
		return data

	def sendTCP(self, data):
		encoded_message = self.encode(data)
		data2send = struct.pack('>I', len(encoded_message)) + encoded_message
		self.clientTCP.send(data2send)

		# time.sleep(self.DELAY)
		# self.send_EOT(conn)

		return

	def sendTCP_SOT(self):
		self.sendTCP(self.START_OF_MESSAGE)
		return
	

	def sendTCP_EOT(self):
		self.sendTCP(self.END_OF_MESSAGE)
		return
	

	def sendUDP(self, data):
		messages = np.array_split(data,  np.arange(self.CHUNK, len(data), self.CHUNK))
		for ind, message in enumerate(messages):
			indexed_message = np.concatenate((np.array([ind * self.CHUNK], dtype=np.float32), message))
			encoded_message = self.encode(indexed_message)
			rnd = np.random.uniform(0,1)
			# if rnd<0.95:
			self.clientUDP.sendto(encoded_message, self.UDP_ADDR)
			time.sleep(self.DELAY)

		return

	# def send(self, cmd, data):
	# 	self.sendTCP_SOT()
	# 	self.sendTCP_CMD(cmd)
	# 	self.sendUDP(data)
	# 	self.sendTCP_EOT()

	# 	return

	def send(self, cmd, msg_udp=None, msg_tcp=None):
		msg_type='tcp'
		if msg_udp is not None and msg_tcp is None:
			msg_type='udp'
		elif msg_udp is not None and msg_tcp is not None:
			msg_type='tcpudp'
			
		self.sendTCP((self.START_OF_MESSAGE,cmd, msg_type, self.CLIENT_ID))
		if msg_type != 'udp':
			self.sendTCP(msg_tcp)
		if msg_type != 'tcp':
			self.sendUDP(msg_udp)
		self.sendTCP((self.END_OF_MESSAGE, cmd, msg_type, self.CLIENT_ID))
		return
	

	def recv_TCP(self, conn):
		raw_data = self.recvall(conn, 4)
		if not raw_data:
			return None
		msg_len = struct.unpack('>I', raw_data)[0]
		msg = self.recvall(conn, msg_len)
		decoded_msg = self.decode(msg)
		return decoded_msg
		
		#tcp receive all raw data
	def recvall(self, conn, data_len):
		data = bytearray()
		while len(data)<data_len:
			packet = conn.recv(data_len - len(data))
			if not packet:
				return None
			data.extend(packet)
		return data
	
	def recv_UDP(self):
		buffer = []
		while True:
			try:
				msg, addr = self.clientUDP.recvfrom(BUFFER)
			except socket.error:
				break

			# if client_id not in self.clients_list.id_list:
			# 	self.clients_list.add_client(client_id, None, addr)

			try:
				decoded_msg = self.decode(msg)
			except:
				continue

			start_index = decoded_msg[0]
			start_index_indices = np.arange(start_index, start_index + len(decoded_msg) - 1, dtype=np.float32)
			start_index_indices_grad = np.vstack([start_index_indices, decoded_msg[1:]]).T
			buffer.append(start_index_indices_grad)
		logging.info('UDP: received {} packets'.format(len(buffer)))
		return buffer	
 
	def receive(self):
		msg = []
		buffer = []
		data = []
		readnext = True
		conn = self.clientTCP
		while readnext:
			try:
				msg = self.recv_TCP(conn)
				if msg is None:
					continue
				flag = msg[0]
				cmd = msg[1]
				msg_type = msg[2]
				# client_id = msg[3]
				# conn.setblocking(False)
			except socket.error:
				pass

			if np.isinf(flag) and np.sign(flag) > 0:
				# conn.setblocking(True)
				readnext = False

			if np.isinf(flag) and np.sign(flag) < 0:
				if msg_type == 'tcp':
					data = self.recv_TCP(conn)
				elif msg_type == 'udp':
					data = self.recv_UDP()
				elif msg_type == 'tcpudp':
					buffer = self.recv_TCP(conn)
					data.append(buffer)
					buffer = self.recv_UDP(conn)
					data.append(buffer)
		return cmd, data
		# self.accumulated_gradient[indices] += gradient


