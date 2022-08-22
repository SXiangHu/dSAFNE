import numpy as np
import networkx as nx
import pandas as pd
import argparse
import torch
import os
import time
import json


def cal_similarity(G, edge, mode='neighbor'):
    weight = 0

    def sim_nei():
        context_from = list(G.neighbors(edge[0]))
        context_from.append(edge[0])
        context_to = list(G.neighbors(edge[1]))
        context_to.append(edge[1])
        num_shared_nei = len(set(context_from).intersection(set(context_to)))
        return num_shared_nei / (G.degree(edge[0]) + 1)

    def sim_de():
        return min(G.degree(edge[0]), G.degree(edge[1])) / max(G.degree(edge[0]),
                                                               G.degree(edge[1]))
    sim_mode = {
        'neighbor': sim_nei,
        'degree': sim_de,
        'both': lambda: 0.5 * (sim_nei() + sim_de())
    }
    assert mode in sim_mode
    return sim_mode[mode]()


#################################################
# parse params
#################################################
def _parse_args(params_config, args):
    parser = argparse.ArgumentParser()
    for params_category in params_config:
        for param_flag, param_config in params_config[params_category].items():
            parser.add_argument(param_flag, **param_config)
    return parser.parse_args(args)


def get_params(params_config, args=None):
    namespace = _parse_args(params_config, args)

    return {
        param_category: {
            param_config['dest']: namespace.__getattribute__(
                param_config['dest']
            ) for param_config in params_config[param_category].values()
        } for param_category in params_config
    }


#################################################
# optimizer & scheduler
#################################################
def _get_grad_requiring_params(models):
    nb_params = 0
    grad_requiring_parmas = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                nb_params += param.numel()
                grad_requiring_parmas.append(param)
        print('{} nb_params={:.2f}M'.format(model.name, nb_params / 1e6))
    return grad_requiring_parmas


def get_optimizer(models, optim, lr, momentum, weight_decay):
    if optim == 'sgd':
        return torch.optim.SGD(_get_grad_requiring_params(models),
                               lr=lr,
                               momentum=momentum)
    elif optim == 'adam':
        return torch.optim.Adam(_get_grad_requiring_params(models),
                                lr=lr,
                                betas=(0.9, 0.99),
                                weight_decay=weight_decay)
    elif optim == 'rmsprop':
        return torch.optim.RMSprop(_get_grad_requiring_params(models),
                                   lr=lr,
                                   alpha=0.9,
                                   weight_decay=weight_decay)


def get_scheduler(optimizer, lr_mode, lr_warmup=0):
    if lr_warmup > 0 and lr_mode == 'raise':
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda ep: min(1, ep / lr_warmup)
        )
    elif lr_mode == 'decay':
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda ep: max(0.33, 0.998**ep)
        )
    return None


#################################################
# checkpoint
#################################################
def _load_checkpoint(checkpoint_path, models, optimizers, schedulers):
    print('loading from a checkpoint at {}.'.format(checkpoint_path))
    checkpoint_state = torch.load(checkpoint_path)
    iter_init = checkpoint_state['iter_no'] + 1  
    models[0].load_state_dict(checkpoint_state['emb_model'])
    models[1].load_state_dict(checkpoint_state['lm_model'])
    for i, optim in enumerate(optimizers):
        optim.load_state_dict(checkpoint_state['optimizer_{}'.format(i)])
    for i, schedu in enumerate(schedulers):
        schedu.step(checkpoint_state['scheduler_{}'.format(i)])
    return iter_init


def load_checkpoint(checkpoint_path, models, optimizers, schedulers):
    if checkpoint_path and os.path.exists(checkpoint_path):
        return _load_checkpoint(checkpoint_path, models,
                                optimizers, schedulers)
    return 0


def save_checkpoint(checkpoint_path, iter_no, models, optimizers, schedulers):
    if checkpoint_path:
        checkpoint_state = {
            'iter_no': iter_no,
            'emb_model': models[0].state_dict(),
            'lm_model': models[1].state_dict(),
        }
        for i, optim in enumerate(optimizers):
            checkpoint_state['optimizer_{}'.format(i)] = optim.state_dict()
        for i, schedu in enumerate(schedulers):
            checkpoint_state['scheduler_{}'.format(i)] = schedu.last_epoch
        torch.save(checkpoint_state, checkpoint_path)


#################################################
# save params when program finished
#################################################
def save_params(log_dir, lm_loss, lm_acc, walk_params,
                model_params, optim_params,
                data_params, trainer_params):
    params = {'walk_params': walk_params,
              'model_params': model_params,
              'optim_params': optim_params,
              'data_params': data_params,
              'trainer_params': trainer_params}
    if lm_loss is not None:
        params['lm_loss'] = lm_loss.item()
    if lm_acc is not None:
        params['lm_acc'] = lm_acc.item()
    with open(os.path.join(log_dir, '%s.json' % time.strftime('%y-%m-%d_%H.%M', time.localtime())), 'w') as f:
        json.dump(params, f)


if __name__ == '__main__':
    g = nx.Graph()
    g.add_nodes_from(['1', '2', '3'])
    g.add_edge('1', '2')
    sim = cal_similarity(g, ['1', '2'])
    print(sim)
