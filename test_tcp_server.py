# Import libraries
import socket
import json
from signal import signal, SIGPIPE, SIG_DFL
# Define IP address and port of Raspberry Pi
server_ip = '10.0.0.12' # Adjust this to match your setup
server_port = 5000

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR , 1)
# Connect to Raspberry Pi
print("Connecting to Raspberry Pi...")

server_socket.bind((server_ip, server_port))
# print("Connected.")
server_socket.listen(5)
signal(SIGPIPE, SIG_DFL)
conn, client_addr = server_socket.accept()

msg_json = conn.recv(1024).decode()

msg = json.loads(msg_json)
print(msg)
input()
msg = 12
msg_json = json.dumps(msg).encode()
server_socket.sendall(msg_json)

while True:
    pass

# Close socket connection
server_socket.close()
