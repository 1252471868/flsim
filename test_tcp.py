# Import libraries
import socket
import json

# Define IP address and port of Raspberry Pi
raspberry_pi_ip = "10.0.0.20" # Adjust this to match your setup
raspberry_pi_port = 5000

# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to Raspberry Pi
print("Connecting to Raspberry Pi...")
client_socket.connect((raspberry_pi_ip, raspberry_pi_port))
print("Connected.")

# Define data to send
data = {
    "name": "Alice",
    "age": 25,
    "hobbies": ["reading", "coding", "gaming"]
}

# Convert data to JSON string
data_json = json.dumps(data)

# Send data to Raspberry Pi
print("Sending data...")
client_socket.send(data_json.encode())
print("Data sent.")

# Close socket connection
client_socket.close()
