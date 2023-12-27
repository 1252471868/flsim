# Import libraries
import cmd
import socket
import socketserver
import json
import struct
from signal import signal, SIGPIPE, SIG_DFL
import dill
from threading import Thread, Lock

# Define IP address and port of Raspberry Pi
server_ip = '192.168.31.12' # Adjust this to match your setup
server_port = 5000
buffersize = 1024

signal(SIGPIPE, SIG_DFL)
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
		self.addr_list.append(client_addr)
		# self.report_list.append(client.Report())

	def get_socket(self, client_id):
		return self.socket_list[self.id_list.index(client_id)]

	def get_addr(self, client_id):
		return self.addr_list[self.id_list.index(client_id)]

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
	

clients_list = ClientList()

class UDPServerHandler(socketserver.BaseRequestHandler):
	def handle(self):
		cmd, data = self.recv_data()
		if cmd == 'ID':
			client_id = data
			clients_list.add_client(client_id, 0, self.client_address)
			print('Client {} connected'.format(client_id))
			print('Address: {}'.format(self.client_address))
		elif cmd == 'ERROR':
			print('error!')
		self.send_data(client_id,'ACK','server received')
   
	def send_data(self, client_id, cmd, data):
		msg = {}
		msg['CMD'] = cmd
		msg['DATA'] = data
		# Convert data to JSON string
		msg_string = dill.dumps(msg)
		# msg_json = jsonpickle.encode(msg).encode()
		# data_len = str(len(msg_json)).encode
		data2send =msg_string
		# logging.info('send datasize: {}'.format(sys.getsizeof(msg_json)))
		client_addr = clients_list.get_addr(client_id)
		# server.sendall(data_len)
		self.request[1].sendto(data2send, client_addr)
  
	def recv_data(self):
		msg = dill.loads(self.request[0])
		cmd = msg['CMD']
		data = msg['DATA']
		return cmd, data




def msg_handler(server_ip,server_port):
	print('Server started, waiting for connecting')
	socketserver.UDPServer.allow_reuse_address = True
	s=socketserver.ThreadingUDPServer((server_ip,server_port), UDPServerHandler)
	s.serve_forever()
	

msg_thread = Thread(target=msg_handler, args=(server_ip,server_port,))
msg_thread.start()
# Create a socket object
# server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR , 1)
# server_socket.bind((server_ip,server_port))
# print('server listening...')
# msg_server='123'
# bytes2send=str.encode(msg_server)
# while(True):
#     bytes_recv = server_socket.recvfrom(buffersize)
#     msg = bytes_recv[0]
#     addr = bytes_recv[1]
#     print('msg : {}'.format(msg))
#     print('addr : {}'.format(addr))
#     server_socket.sendto(bytes2send,addr)
# Connect to Raspberry Pi
# print("Connecting to Raspberry Pi...")




while True:
	pass

# Close socket connection
server_socket.close()
