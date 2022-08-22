import numpy as np
import os
import pandas as pd
import networkx as nx
import json
import torch
import scipy.io as sio


class Data(object):
    def __init__(self, dataset, data_format):
        self.dataset = dataset
        self.format = data_format

    def load_data(self, data_path):
        print('loading {}/{} dataset...'.format(data_path, self.dataset))
        G = None
        labels = None
        attrs = None
        if self.format == 'gml':
            G = nx.read_gml(os.path.join(data_path, '{}.{}'.format(self.dataset, self.format)))
            node_value = G._node
            labels = list(map(lambda x: x[1].get('value', -1), node_value.items()))
        elif self.format == 'twitter':
            data_path = os.path.join(data_path, 'twitter/%s' % self.dataset)
            edges = np.genfromtxt(os.path.join(data_path,
                                               '{}.cites'.format(self.dataset)),
                                  dtype=np.dtype(str))
            file_list = os.listdir(data_path)
            feat_names = None
            nodes = None
            ego_nets = {}
            for file in file_list:
                ex = os.path.splitext(file)
                ego = ex[0]
                if ex[1] == '.content':
                    nodes_info = np.genfromtxt(os.path.join(data_path, file),
                                               encoding='utf-8',
                                               dtype=np.dtype(str))
                    ego_nets[ego] = {
                        node[0]: {
                            'attr': node[1: -1].astype(float),
                            'label': node[-1]
                        } for node in nodes_info
                    }
                    if nodes is None:
                        nodes = nodes_info[:, 0]
                    else:
                        nodes = np.concatenate([nodes, nodes_info[:, 0]])
                elif ex[1] == '.featnames':
                    ego_featnames = []
                    with open(os.path.join(data_path, file), 'r', encoding='utf-8') as f:
                        while True:
                            featname = f.readline()
                            if featname:
                                ego_featnames.append(str.strip(featname.split(' ')[-1]))
                                print(featname)
                            else:
                                break
                    print('ego {}, featnames_len: {}'.format(
                        ego, len(ego_featnames)
                    ))
                    ego_nets[ego]['featnames'] = ego_featnames.copy()
                    if feat_names is None:
                        feat_names = ego_featnames
                    else:
                        feat_names += ego_featnames
            nodes = np.unique(nodes)
            labels = np.zeros_like(nodes).astype(str).tolist()
            nodes = nodes.tolist()
            feat_names = np.unique(feat_names)
            attrs = pd.DataFrame(np.zeros([len(nodes), len(feat_names)]),
                                 index=nodes,
                                 columns=feat_names)
            for ego, ego_net in ego_nets.items():
                for node, info in ego_net.items():
                    print(ego, node)
                    if node == 'featnames':
                        continue
                    attrs.loc[node, ego_net['featnames']] = info['attr']
                    labels[nodes.index(node)] = info['label']
            G = nx.Graph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)
            attrs = attrs.values
        elif self.format == 'cora':
            idx_features_labels = np.genfromtxt(os.path.join(data_path,
                                                             '{}.content'.format(self.dataset)),
                                                dtype=np.dtype(str))
            edges = np.genfromtxt(os.path.join(data_path,
                                               '{}.cites'.format(self.dataset)),
                                  dtype=np.dtype(str))
            nodes = idx_features_labels[:, 0]

            err = list(set(edges.reshape(-1).tolist()).difference(set(nodes.tolist())))
            index = []
            for i, edge in enumerate(edges):
                for node in edge:
                    if node in err:
                        index.append(i)
                        print(index)
                        continue

            edges = np.delete(edges, index, axis=0)

            labels = idx_features_labels[:, -1]
            attrs = idx_features_labels[:, 1:-1].astype(float)
            G = nx.Graph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)
        elif self.format == 'mat':
            mat = sio.loadmat(os.path.join(data_path, '{}.{}'.format(
                self.dataset, self.format
            )))
            network = mat['network']
            G = nx.from_scipy_sparse_matrix(network)
            labels = np.argmax(mat['group'].toarray(), axis=-1)
        elif self.format == 'blogCatalog':
            G = nx.Graph()
            with open(os.path.join(data_path,
                                   'blogCatalog/bc_edgelist.txt'),
                      'r') as p:
                while 1:
                    edge = p.readline()
                    if edge == '':
                        break
                    src, dst = edge.strip().split()
                    G.add_edge(src, dst)
                    G.add_edge(dst, src)
            labels = pd.DataFrame(np.zeros([len(G.nodes), 39], dtype=int),
                                  index=list(G.nodes),
                                  columns=[str(i) for i in range(39)])
            with open(os.path.join(data_path,
                                   'blogCatalog/bc_labels.txt'),
                      'r') as p:
                while 1:
                    l = p.readline()
                    if l == '':
                        break
                    l = l.strip().split()
                    labels.loc[l[0], l[1:]] = 1
            labels = labels.values
        else:
            raise Exception('dataset {} format error'.format(self.dataset))
        labels = self.str2num(labels)
        return G, labels, attrs

    @staticmethod
    def str2num(labels):
        str_num_map = {label: i for i, label in enumerate(set(labels))}
        num_labels = list(map(lambda x: str_num_map[x], labels))
        return num_labels
    
    @staticmethod
    def vocab2int(nodes):
        vocab_2_int = {n: i for i, n in enumerate(nodes)}
        return list(map(lambda n: vocab_2_int[n], nodes))


class Corpus:
    def __init__(self, corpus_dir, corpus_name, window_size, p, q, walk_mode):
        assert os.path.exists(corpus_dir)
        corpus_path = os.path.join(corpus_dir, f'corpus_{window_size}_p{p}_q{q}_{walk_mode}.json')
        vocab_path = os.path.join(corpus_dir, 'vocab.json')
        adj_path = os.path.join(corpus_dir, 'adj.npy')
        attr_path = os.path.join(corpus_dir, 'attrs.npy')
        attrs_sim_path = os.path.join(corpus_dir, 'attrs_sim.npy')
        with open(corpus_path, 'r') as f:
            corpus_raw = np.array(json.load(f))
        with open(vocab_path, 'r') as f:
            self._dictionary = json.load(f)
        self._attr = None
        self._attrs_sim = None
        self._adj = torch.Tensor(np.load(adj_path))
        if os.path.exists(attr_path):
            self._attr = torch.Tensor(np.load(attr_path).astype(np.int))
            self._attrs_sim = torch.Tensor(
                np.load(attrs_sim_path).astype(np.float)
            )

        self.train = _tokenize(corpus_raw, corpus_name, self._dictionary)

    @property
    def vocab_size(self):
        return len(self._dictionary)

    @property
    def vocab(self):
        return self._dictionary

    @property
    def adj(self):
        return self._adj

    @property
    def attr(self):
        return self._attr

    @property
    def attrs_sim(self):
        return self._attrs_sim


def batchify(data_tensor, batch_size):
    nb_batches = data_tensor.size(0) // batch_size
    data_tensor = data_tensor.narrow(0, 0, nb_batches * batch_size)
    data_tensor = data_tensor[torch.randperm(nb_batches * batch_size)]
    data_tensor = data_tensor.view(-1, batch_size, data_tensor.size(-1)).contiguous()
    return data_tensor


def _build_corpus(corpus_dir, dataset, window_size, p, q, walk_mode):
    corpus_path = os.path.join(corpus_dir, f'corpus_{window_size}_p{p}_q{q}_{walk_mode}.pt')
    if os.path.exists(corpus_path):
        print('Loading an existing corpus file from {}'.format(corpus_path))
        corpus = torch.load(corpus_path)
    else:
        print('Creating a corpus file at {}'.format(corpus_path))
        corpus = Corpus(corpus_dir, dataset, window_size, p, q, walk_mode)
        torch.save(corpus, corpus_path)
    return corpus


def _tokenize(corpus_raw, corpus_name, dictionary):
    print('Tokenizing {}'.format(corpus_name))
    ids = []
    nums = 0
    for tokens in corpus_raw:
        id = list(map(lambda x: dictionary[str(x)], tokens))
        if len(id) != 100:
            nums += 1
            continue
        ids.append(id)
    ids = torch.Tensor(ids)
    return ids


def _get_train_data(corpus, batch_size):
    return batchify(corpus.train, batch_size)


def get_train_data(data_params, window_size, p, q, walk_mode, batch_size, device):
    corpus = _build_corpus(data_params['corpus_dir'],
                           data_params['dataset'],
                           window_size,
                           p,
                           q,
                           walk_mode)
    data_params['vocab_size'] = corpus.vocab_size
    # train_data = _get_train_data(corpus, batch_size)
    # train_data = train_data
    return corpus
