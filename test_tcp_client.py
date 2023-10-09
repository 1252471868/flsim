# Import libraries
import socket
import json

# Define IP address and port of Raspberry Pi
server_ip = '10.0.0.12' # Adjust this to match your setup
server_port = 5000

# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.setblocking(1)
# Connect to Raspberry Pi
client_socket.connect((server_ip,server_port))

msg = 12
msg_json = json.dumps(msg).encode()
client_socket.sendall(msg_json)
while True:
    data_len_json = client_socket.recv(1024).decode()
    print('123')
# msg_json = client_socket.recv(1024).decode()

# msg = json.loads(msg_json)
print(msg)

# Close socket connection
client_socket.close()
