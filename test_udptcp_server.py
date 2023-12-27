import TCPUDP_socket
import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

server_ip = '192.168.1.12'

server_socket = TCPUDP_socket.TCPUDPServer(SERVER=server_ip,NUM_CLIENTS=1)
server_socket.start(1)
while len(server_socket.clients_list.id_list) < 1 :
        pass
server_socket.send(0, 'INFO', msg_tcp=11)
server_socket.send(0, 'INFO', msg_udp=[12])

input('Press a key to end: ')
server_socket.send(0, 'END', msg_tcp=11)