import argparse
import client
import config
import logging
import os
import server
import time

# server_config = './configs/MNIST/mnist_marl_train_noniid.json'
server_config = './configs/MNIST/mnist_marl_eval_noniid.json'
# server_config = './configs/MNIST/mnist_fedavg_noniid.json'
# server_config = './configs/FashionMNIST/fmnist_fedavg_noniid.json'
# server_config = './configs/CIFAR-10/cifar_fedavg_noniid.json'
# server_config = './config.json'
# Set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default=server_config,
                    help='Federated learning configuration file.')
# parser.add_argument('-c', '--config', type=str, default='./configs/MNIST/mnist.json',
#                     help='Federated learning configuration file.')
parser.add_argument('-l', '--log', type=str, default='INFO',
                    help='Log messages level.')

args = parser.parse_args()
case_name = args.config.split('/')[-1].split('.')[0]
print("case_name:", case_name)

# Set logging
logging.basicConfig(
    format='[%(levelname)s][%(asctime)s]: %(message)s', level=getattr(logging, args.log.upper()), datefmt='%H:%M:%S')


def main():
    """Run a federated learning simulation."""

    # Read configuration file
    fl_config = config.Config(args.config)

    # Initialize server
    fl_server = {
        "basic": server.Server(fl_config, case_name),
        "accavg": server.AccAvgServer(fl_config, case_name),
        "directed": server.DirectedServer(fl_config, case_name),
        "kcenter": server.KCenterServer(fl_config, case_name),
        "kmeans": server.KMeansServer(fl_config, case_name),
        "magavg": server.MagAvgServer(fl_config, case_name),
        "dqn": server.DQNServer(fl_config, case_name), # DQN inference server 
        "dqntrain": server.DQNTrainServer(fl_config, case_name), # DQN train server
        "marl": server.MARLTrainServer(fl_config, case_name), # DQN train server
    }[fl_config.server.type]
    fl_server.boot()

    # Run federated learning
    t_start = time.time()
    fl_server.run()
    t_end = time.time()
    logging.info('Running time: {}'.format(t_end-t_start))
    # Delete global model
    # os.remove(fl_config.paths.model + '/global')


if __name__ == "__main__":
    main()
