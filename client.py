import logging
import torch
import torch.nn as nn
import torch.optim as optim
import socket
import json
import random
import utils.dists as dists  # pylint: disable=no-name-in-module

class Client(object):
    """Simulated federated learning client."""

    def __init__(self, client_id):
        self.client_id = client_id
        
    def __repr__(self):
        return 'Client #{}: {} samples in labels: {}'.format(
            self.client_id, len(self.data), set([label for _, label in self.data]))
    # Set up client
    def boot(self, config):
        # logging.info('Booting {} server...'.format(self.config.server))
        self.config = config
        self.connect_server()

# TCP/IP connect to server
    def connect_server(self):
        server_ip = self.config.server.socket.get('ip')
        server_port = self.config.server.socket.get('port')
        self.server_addr = (server_ip,server_port)
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        logging.info('Connecting to server')
        self.client_socket.connect(self.server_addr)
        logging.info('Server connected')
        self.send_data(self.client_socket, 'ID', self.client_id)
        

    # Set non-IID data configurations
    def set_bias(self, pref, bias):
        self.pref = pref
        self.bias = bias

    def set_shard(self, shard):
        self.shard = shard
    def set_socket(self, client_socket):
        self.client_socket = client_socket
        
    # # Server interactions
    # def download(self, argv):
    #     # Download from the server.
    #     if self.socket_state == False:
    #         try:
    #             return argv.copy()
    #         except:
    #             return argv
    #     # else:
            

    # def upload(self, argv):
    #     # Upload to the server
    #     try:
    #         return argv.copy()
    #     except:
    #         return argv

    # Federated learning phases
    def set_data(self, data):
        # Extract from config
        do_test = self.do_test = self.config.clients.do_test
        test_partition = self.test_partition = self.config.clients.test_partition

        # Download data
        self.data = data
        # cmd, self.data = self.recv_data()

        # Extract trainset, testset (if applicable)
        if do_test:  # Partition for testset if applicable
            self.trainset = data[:int(len(data) * (1 - test_partition))]
            self.testset = data[int(len(data) * (1 - test_partition)):]
        else:
            self.trainset = data

    def configure(self, model):
        import fl_model  # pylint: disable=import-error

        # Extract from config
        model_path = self.model_path = self.config.paths.model

        # Download from server

        # Extract machine learning task from config
        self.task = self.config.fl.task
        self.epochs = self.config.fl.epochs
        self.batch_size = self.config.fl.batch_size

        # Download most recent global model
        path = model_path + '/global'
        self.model = model
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

        # Create optimizer
        self.optimizer = fl_model.get_optimizer(self.model)

    def run(self):
        while True:
            cmd, data = self.recv_data()
            if cmd == 'CONFIG':
                self.set_data(data)
            elif cmd == 'MODEL':
                self.configure(data)
                {
                    "train": self.train()
                }[self.task]
        # Perform federated learning task
        {
            "train": self.train()
        }[self.task]

    # def get_report(self):
    #     # Report results to server.
    #     return self.upload(self.report)

    # Machine learning tasks
    def train(self):
        import fl_model  # pylint: disable=import-error

        logging.info('Training on client #{}'.format(self.client_id))

        # Perform model training
        trainloader = fl_model.get_trainloader(self.trainset, self.batch_size)
        fl_model.train(self.model, trainloader,
                       self.optimizer, self.epochs)

        # Extract model weights and biases
        weights = fl_model.extract_weights(self.model)
        
        # Generate report for server
        self.report = Report(self)
        self.report.weights = weights
        # Perform model testing if applicable
        if self.do_test:
            testloader = fl_model.get_testloader(self.testset, 1000)
            self.report.accuracy = fl_model.test(self.model, testloader)
        self.send_data(self.client_socket, 'REPORT', self.report)

    def test(self):
        # Perform model testing
        raise NotImplementedError
    
    def send_data(self, server, cmd, data):
        msg = {}
        msg['CMD'] = cmd
        msg['DATA'] = data
        # Convert data to JSON string
        msg_json = json.dumps(msg)
        server.sendall(msg_json.encode())

    def recv_data(self):
        msg_json = self.client_socket.recv(1024).decode()
        msg = json.loads(msg_json)
        cmd = msg['CMD']
        data = msg['DATA']
        return cmd, data





class Report(object):
    """Federated learning client report."""

    def __init__(self, client):
        self.client_id = client.client_id
        self.num_samples = len(client.data)
