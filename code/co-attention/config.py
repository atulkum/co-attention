import os

class Config(object):
    pass

config = Config()
config.data_dir = os.path.join(os.path.expanduser('~'), 'co-attention/data')
config.log_root = os.path.join(os.path.expanduser('~'), 'co-attention/log')
config.embedding_path = os.path.join(config.data_dir, 'glove.trimmed.100.npz')

config.context_len = 600
config.question_len = 30

config.hidden_dim = 200
config.embedding_size=100

#vector with zeros for unknown words
config.max_dec_steps = 4
config.maxout_pool_size=16

config.lr = 0.001
config.dropout_ratio = 0.15

config.max_grad_norm = 5.0
config.batch_size = 100
config.num_epochs = 50

config.print_every = 100
config.save_every = 50000000
config.eval_every = 1000

config.model_type = 'co-attention'
