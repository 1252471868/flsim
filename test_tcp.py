# Import libraries
import socket
import json

# Define IP address and port of Raspberry Pi
server_ip = '10.0.0.13' # Adjust this to match your setup
server_port = 5000

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to Raspberry Pi
print("Connecting to Raspberry Pi...")
server_socket.bind((server_ip, server_port))
# print("Connected.")
server_socket.listen(5)

conn, client_addr = server_socket.accept()

msg_json = conn.recv(1024).decode()

msg = json.loads(msg_json)
print(msg)

# Close socket connection
server_socket.close()
