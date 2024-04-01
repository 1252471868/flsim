import socket

def receive_udp_packet(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))
    
    while True:
        data, addr = sock.recvfrom(1024)
        message = data.decode()
        print(f'Received message from {addr}: {message}')

# Example usage
local_ip = '192.168.2.12'  # Listen on all available network interfaces
local_port = 5050  # Replace with the desired port number

receive_udp_packet(local_ip, local_port)