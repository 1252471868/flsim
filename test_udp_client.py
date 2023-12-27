# Import libraries
import socket
import json
from signal import signal, SIGPIPE, SIG_DFL
import dill
import json
import struct
# Define IP address and port of Raspberry Pi
server_ip = '192.168.31.12' # Adjust this to match your setup
server_port = 5000
server_addr = (server_ip, server_port)
buffersize = 1024
        
def recv_data(self):
    # datalength : 4 bytes
    raw_data = self.recvall(4)
    if not raw_data:
        return None, None
    msg_len = struct.unpack('>I', raw_data)[0]

    msg_string = self.recvall(msg_len)

    msg = dill.loads(msg_string)

    cmd = msg['CMD']
    data = msg['DATA']
    return cmd, data

def send_data(server, cmd, data):
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
#receive all raw data
def recvall(self, data_len):
    data = bytearray()
    while len(data)<data_len:
        packet = self.request.recv(data_len - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data
# Create a socket object
signal(SIGPIPE, SIG_DFL)
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
msg = {}
msg['CMD'] = 'ID'
msg['DATA'] = 0

msg2send = dill.dumps(msg)
client_socket.sendto(msg2send, server_addr)
print('connected')

msg_recv = client_socket.recvfrom(buffersize)
data = dill.loads(msg_recv[0])

print('cmd from server: {}'.format(data['CMD']))
print('data from server: {}'.format(data['DATA']))

# msg = 12
# msg_json = json.dumps(msg).encode()
# client_socket.sendall(msg_json)
# while True:
#     data_len_json = client_socket.recv(1024).decode()
#     print('123')
# msg_json = client_socket.recv(1024).decode()

# msg = json.loads(msg_json)
# print(msg)

# Close socket connection
# client_socket.close()
