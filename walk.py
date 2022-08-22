import numpy as np
import networkx as nx
import random
import sys
import os
from dataset import Data
import torch
from config import PARAMS_CONFIG
from man_utils.utils import (
    get_params,
    cal_similarity,
)
import pandas as pd
import json
from multiprocessing import Pool


class Walk():
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = float(p)
        self.q = float(q)

    def walk(self, walk_length, start_node, window_size):
        G = self.G
        alias_nodes = self.alias_nodes

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0],
                                                    alias_nodes[cur][1])])
                else:
                    dws = self.cal_dynamic_weight(
                        walk[-(min(window_size, len(walk))):], cur_nbrs)
                    next = random.choices(cur_nbrs, dws)
                    walk += next
            else:
                break
        return walk

    def cal_dynamic_weight(self, pres, nbrs):
        G = self.G
        dws = []
        for nbr in nbrs:
            weight = G[pres[-1]][nbr]['weight']
            for i in range(1, len(pres)):
                weight += (self.q**i) * \
                    (G[pres[-1 - i]][nbr]['weight']
                     if nbr in G[pres[-1 - i]] else 0)
            weight /= (self.p if nbr == pres[-2] else 1)
            dws.append(weight)
        normalized_dws = normalize(dws)
        return normalized_dws

    def simulate_walks(self, num_walks, walk_length, window_size):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            print(f'\nWalks per node:{walk_iter + 1}/{num_walks}')
            random.shuffle(nodes)
            # for node in nodes:
            for j, node in enumerate(nodes):
                sys.stdout.write(f'\r{j + 1}/{len(nodes)}')
                sys.stdout.flush()
                walks.append(
                    self.walk(walk_length=walk_length,
                              start_node=node,
                              window_size=window_size))

        return walks

    def cal_alias_nodes(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G
        # is_directed = self.is_directed
        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [
                G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))
            ]
            normalized_probs = normalize(unnormalized_probs)
            alias_nodes[node] = alias_setup(normalized_probs)

        self.alias_nodes = alias_nodes
        return


def normalize(unnormalized_probs):
    norm_const = sum(unnormalized_probs)
    normalized_probs = [
        float(u_prob) / norm_const for u_prob in unnormalized_probs
    ]
    return normalized_probs


def alias_setup(probs):
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def cal_attr_sim(attrs_block, attrs, attrs_sim, start_pos):
    for i, attr in enumerate(attrs_block):
        print(f'{os.getpid()} cal attr_sim {i + 1}/{attrs_block.size(0)}.')
        attr_sim = torch.cosine_similarity(attr, attrs, dim=-1)
        attrs_sim[start_pos + i] = attr_sim


def cal_weight(edges, G, walk_mode):
    for i, edge in enumerate(edges):
        print(f'process: {os.getpid()} cal edge weight {i + 1}/{len(edges)}')
        # attrs_sim[edge[0], edge[1]] = G[edge[0]][edge[1]]['weight']
        if 'weight' in G[edge[0]][edge[1]].keys():
            G[edge[0]][edge[1]]['weight'] *= cal_similarity(
                G, edge, walk_mode)
        else:
            G[edge[0]][edge[1]]['weight'] = cal_similarity(
                G, edge, mode=walk_mode)


def main(walk_params, data_params, **kwargs):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    output_dir = f'./data/train/{data_params["dataset"]}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    input_dir = './data'

    data = Data(data_params['dataset'], data_params['data_format'])
    G, label, attrs = data.load_data(input_dir)
    vocab = list(G.nodes())
    mat = nx.to_scipy_sparse_matrix(G)
    adj_ = mat.toarray()
    np.save(os.path.join(output_dir, 'adj_noweight.npy'), adj_)
    vocab_2_int = {c: i for i, c in enumerate(vocab)}
    print('start walking...')
  
    edges = list(G.edges())
    cal_weight(edges, G, walk_params['walk_mode'])
    adj = pd.DataFrame(adj_, index=vocab, columns=vocab)
    
    G_ = Walk(G, False, walk_params['p'], walk_params['q'])
    G_.cal_alias_nodes()
    corpus = G_.simulate_walks(walk_params['walk_num'],
                               walk_params['path_length'],
                               walk_params['window_size'])

    with open(os.path.join(output_dir, f'corpus_{walk_params["window_size"]}_p{walk_params["p"]}_q{walk_params["q"]}_{walk_params["walk_mode"]}.json'), 'w') as f:
        json.dump(corpus, f)

if __name__ == '__main__':
    main(**get_params(params_config=PARAMS_CONFIG))
