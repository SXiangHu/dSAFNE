PARAMS_CONFIG = {
    'walk_params': {
        '--walk_num': {
            'default': 100,
            'type': int,
            'help': 'walk num per vertex',
            'dest': 'walk_num'
        },
        '--path_length': {
            'default': 100,
            'type': int,
            'help': 'walk path length',
            'dest': 'path_length'
        },
        '--walk_mode': {
            'default': 'neighbor',
            'help': '',
            'dest': 'walk_mode'
        },
        '--adj_weight_mode': {
            'default': 'neighbor',
            'help': '',
            'dest': 'adj_weight_mode'
        },
        '--adj_weight_threshold': {
            'default': 0.3,
            'help': '',
            'dest': 'adj_weight_threshold'
        },
        '--window_size': {
            'default': 4,
            'type': int,
            'help': '',
            'dest': 'window_size'
        },
        '--p': {
            'default': 0.2,
            'type': int,
            'help': 'beta',
            'dest': 'p'
        },
        '--q': {
            'default': 0.5,
            'help': 'alpha',
            'dest': 'q'
        }
    },
    'model_params': {
        '--bi_flag': {
            'default': 1,
            'type': int,
            'help': '',
            'dest': 'bi_flag'
        },
        '--dropout': {
            'default': 0.1,
            'type': float,
            'dest': 'dropout'
        },
        '--embedding_dim': {
            'type': int,
            'default': 64,
            'dest': 'embedding_dim'
        },
        '--heads_num': {
            'type': int,
            'default': 1,
            'dest': 'heads_num'
        },
        '--hidden_dim': {
            'type': int,
            'default': 512,
            'dest': 'hidden_dim'
        },
        '--layer_num': {
            'type': int,
            'default': 18,
            'dest': 'layer_num'
        }
    },
    'optim_params': {
        '--lm_optim': {
            'default': 'adam',
            'dest': 'lm_optim'
        },
        '--lm_lr': {
            'default': 0.001,
            'type': float,
            'help': '',
            'dest': 'lm_lr'
        },
        '--dis_optim': {
            'default': 'rmsprop',
            'dest': 'dis_optim'
        },
        '--dis_lr': {
            'default': 0.001,
            'type': float,
            'help': '',
            'dest': 'dis_lr'
        },
        '--grad_clip': {
            'default': 5.,
            'type': float,
            'dest': 'grad_clip'
        },
        '--momentum': {
            'default': 0.5,
            'type': float,
            'dest': 'momentum'
        },
        '--lm_lr_warmup': {
            'default': 1000,
            'type': int,
            'dest': 'lm_lr_warmup'
        },
        '--weight_decay': {
            'default': 1e-8,
            'type': float,
            'dest': 'weight_decay'
        }
    },
    'data_params': {
        '--format': {
            'default': 'cora',
            'help': 'the data format',
            'dest': 'data_format'
        },
        '--dataset': {
            'default': 'cora',
            'help': 'the data name',
            'dest': 'dataset'
        },
    },
    'trainer_params': {
        '--hyper_param': {
            'default': 'iters',
            'type': str,
            'dest': 'hyper_param',
            'help': ''
        },
        '--batch_size': {
            'default': 128,
            'type': int,
            'dest': 'batch_size'
        },
        '--batch_per_iter': {
            'default': 300,
            'type': int,
            'dest': 'batch_per_iter'
        },
        '--iters': {
            'default': 40,
            'type': int,
            'dest': 'iters'
        },
        '--lm_iters': {
            'default': 1,
            'type': int,
            'help': '',
            'dest': 'lm_iters'
        },
        '--dis_iters': {
            'default': 3,
            'type': int,
            'help': '',
            'dest': 'dis_iters'
        },
        '--alpha': {
            'default': 1.0,
            'type': float,
            'help': 'lm l2 reg',
            'dest': 'alpha'
        },
        '--beta': {
            'default': 1.0,
            'type': float,
            'help': 'dis l2 reg',
            'dest': 'beta'
        },
        '--seed': {
            'default': 42,
            'dest': 'seed'
        },
        '--is_attr': {
            'default': 1,
            'type': int,
            'dest': 'is_attr'
        },
        '--info_mode': {
            'default': 'cat',
            'help': '',
            'dest': 'info_mode'
        },
        '--block_size': {
            'default': 128,
            'type': int,
            'dest': 'block_size'
        },
        '--dis_random': {
            'default': 0,
            'type': int,
            'dest': 'dis_ramdom'
        },
        '--dis_loss_mode': {
            'default': 'dis',
            'type': str,
            'dest': 'dis_loss_mode'
        },
        '--k': {
            'default': 20,
            'type': int,
            'dest': 'k'
        }
    }
}