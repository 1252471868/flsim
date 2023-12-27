import TCPUDP_socket
import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

server_ip = '192.168.1.12'
client_socket = TCPUDP_socket.TCPUDPClient(SERVER= server_ip, CLIENT_ID=0)

id = int(input('Input client ID: '))
# client_socket.send('INFO', msg_udp=[0])
client_socket.send('ID', msg_tcp=id, msg_udp=[id])

while True:
    # cmd, data = self.recv_data()
    cmd, data = client_socket.receive()
    if cmd == 'MODEL':
        logging.info('Received model')

    elif cmd == 'INFO':
        logging.info('{}'.format(data))
    elif cmd == 'END':
        logging.info('Completed.')
        # self.client_socket.close()
        break
    
