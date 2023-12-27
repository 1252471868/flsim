import argparse
import client
import config
import logging
import os


# Set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='./config.json',
                    help='Federated learning configuration file.')
# parser.add_argument('-c', '--config', type=str, default='./configs/MNIST/mnist.json',
#                     help='Federated learning configuration file.')
parser.add_argument('-l', '--log', type=str, default='INFO',
                    help='Log messages level.')

args = parser.parse_args()

# Set logging
logging.basicConfig(
    format='[%(levelname)s][%(asctime)s]: %(message)s', level=getattr(logging, args.log.upper()), datefmt='%H:%M:%S')


def main():
    """Run a federated learning simulation."""

    # Read configuration file
    fl_config = config.Config(args.config)
    id = int(input('Input client ID: '))
    fl_client = client.Client(id)
    fl_client.boot(fl_config)
    # Initialize server
    # Run federated learning
    fl_client.run()

    # Delete global model
    # os.remove(fl_config.paths.model + '/global')


if __name__ == "__main__":
    main()
