import argparse
import config
import logging
import os
import pickle as pk
from sklearn.decomposition import PCA
import server
import time
import numpy as np

pca_config = './configs/MNIST/mnist_pca.json'
# Set logging
logging.basicConfig(
    format='[%(levelname)s][%(asctime)s]: %(message)s', level=logging.INFO, datefmt='%H:%M:%S')

# Set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default=pca_config,
                    help='Configuration file for server.')
parser.add_argument('-o', '--output', type=str, default='./output.pkl',
                    help='Output pickle file')

args = parser.parse_args()
case_name = args.config.split('/')[-1].split('.')[0]
print("case_name:", case_name)

def main():
    """Extract PCA vectors from FL clients."""

    # Read configuration file
    fl_config = config.Config(args.config)

    # Initialize server
    fl_server = server.KMeansServer(fl_config, case_name)
    fl_server.boot()

    # Run client profiling
    fl_server.profile_clients()

    # group = [fl_server.clients[profile] for profile in fl_server.clients.keys()]
    # Extract clients, reports, weights
    # clients = [client for client in group]
    clients = [client for profile in fl_server.clients.keys() for client in fl_server.clients[profile]] 
    
    reports = fl_server.reporting(clients)
    # reports = [client.get_report() for client in clients]
    weights = [report.weights for report in reports]

    clients_weights = [fl_server.flatten_weights(report.weights) for report in reports] # list of numpy arrays
    clients_weights = np.array(clients_weights) # convert to numpy array
    clients_prefs = [report.pref for report in reports] # dominant class in each client

    t_start = time.time()
    print("Start building the PCA transformer...")
    pca_n_components = fl_config.clients.total
    pca = PCA(n_components=pca_n_components)
    #self.pca = PCA(n_components=2)
    clients_weights_pca = pca.fit_transform(clients_weights)

    # dump clients_weights_pca out to pkl file for plotting
    clients_weights_pca_fn = 'output/clients_weights_pca.pkl'
    pk.dump(clients_weights_pca, open(clients_weights_pca_fn,"wb"))
    print("clients_weights_pca dumped to", clients_weights_pca_fn)    

    # dump clients_prefs
    clients_prefs_fn = 'output/clients_prefs.pkl'
    pk.dump(clients_prefs, open(clients_prefs_fn,"wb"))
    print("clients_prefs dumped to", clients_prefs_fn)        

    t_end = time.time()
    print("Built PCA transformer, time: {:.2f} s".format(t_end - t_start))


    # # Flatten weights
    # def flatten_weights(weights):
    #     weight_vecs = []
    #     for _, weight in weights:
    #         weight_vecs.extend(weight.flatten())
    #     return weight_vecs

    # logging.info('Flattening weights...')
    # weight_vecs = [flatten_weights(weight) for weight in weights]

    # # Perform PCA on weight vectors
    # logging.info('Assembling output...')
    # output = [(clients[i].client_id, clients[i].pref, weight) for i, weight in enumerate(weight_vecs)]
    # logging.info('Writing output to binary...')
    # with open(args.output, 'wb') as f:
    #     pickle.dump(output, f)

    logging.info('Done!')

if __name__ == "__main__":
    main()
