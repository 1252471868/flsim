from threading import Thread

def msg_handler(server_ip,server_port):
    # logging.info('Server start, waiting for connecting')
    print(server_ip,server_port)


server_ip = '1.1.1.1'
server_port = 5000
msg_thread = Thread(target=msg_handler, args=[server_ip,server_port])
msg_thread.start()