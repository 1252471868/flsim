from collections import namedtuple
import json


class Config(object):
    """Configuration module."""

    def __init__(self, config):
        self.paths = ""
        # Load config file
        with open(config, 'r') as config:
            self.config = json.load(config)
        # Extract configuration
        self.extract()

    def extract(self):
        config = self.config

        # -- Clients --
        '''
        label_distribution: uniform, normal
        '''
        fields = ['total', 'per_round', 'label_distribution',
                  'do_test', 'test_partition','socket']
        defaults = (0, 0, 'uniform', False, None, None)
        params = [config['clients'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.clients = namedtuple('clients', fields)(*params)
        assert self.clients.per_round <= self.clients.total

        # -- Data --
        '''
        loading: static, dynamic
        '''
        fields = ['loading', 'partition', 'IID', 'bias', 'shard']
        defaults = ('static', 0, True, None, None)
        params = [config['data'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.data = namedtuple('data', fields)(*params)

        # Determine correct data loader
        assert self.data.IID ^ bool(self.data.bias) ^ bool(self.data.shard)
        if self.data.IID:
            self.loader = 'basic'
        elif self.data.bias:
            self.loader = 'bias'
        elif self.data.shard:
            self.loader = 'shard'

        # -- Federated learning --
        fields = ['rounds', 'target_accuracy', 'task', 'epochs', 'batch_size']
        defaults = (0, None, 'train', 0, 0)
        params = [config['federated_learning'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.fl = namedtuple('fl', fields)(*params)

        # -- Model --
        self.model = config['model']

        # -- Paths --
        fields = ['data', 'model', 'reports']
        defaults = ('./data', './models', None)
        params = [config['paths'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        # Set specific model path
        params[fields.index('model')] += '/' + self.model
        self.paths = namedtuple('paths', fields)(*params)

        # -- Server --
        fields = ['type', 'socket']
        defaults = ('basic', None)
        params = [config['server'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.server = namedtuple('server', fields)(*params)

        # -- DQN --
        fields = ['episode', 'max_steps', 'learning_rate', 'gamma', 'epsilon_initial', 
                  'epsilon_decay', 'epsilon_min', 'memory_size', 'batch_size', 'target_update',
                  'hidden_layers', 'reward_fun', 'reward_xi', 'rewards_log', 'saved_model', 'trained_model',
                  'pca_model']

        defaults = (200, 300, 0.01, 0.95, 1.0, 0.95, 0.01, 200, 4, 10, 
                    [512,512], 'target', 64, 
                    'output/dqn_trained_rewards.csv',
                    '/output/dqn_trained_model',
                    "output/dqn_models/dqn_10-100_model_200.h5",
                    'output/dqn_models/pca_10_100.pkl'
                    )
        params = [config['dqn'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.dqn = namedtuple('dqn', fields)(*params)