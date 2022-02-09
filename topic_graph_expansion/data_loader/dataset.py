import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import dgl
import networkx as nx
from networkx.algorithms import ancestors, descendants
from gensim.models import KeyedVectors

import os
from tqdm import tqdm
from itertools import chain, product
from collections import deque

import time
import random
import pickle


class Taxo(object):
    def __init__(self, taxo_id, name="None", display_name="None", rank=-1, level=-100, p_count=0, c_count=0):
        self.taxo_id = taxo_id
        self.name = name
        self.display_name = display_name
        self.rank = int(rank)
        self.level = int(level)
        self.p_count = int(p_count)
        self.c_count = int(c_count)


class AIDataset(object):
    #def __init__(self, taxonomy_name, dataset_path, load_file=True, partitioned=False, sample_position='internal'):
    def __init__(self, taxonomy_name, dataset_path, load_file=True, partitioned=False):
        self.taxonomy_name = taxonomy_name  # taxonomy name
        self.partitioned = partitioned
        #self.sample_position = sample_position

        self.full_graph = dgl.DGLGraph()  # full graph (including masked train/validation node indices)
        self.train_node_ids = []
        self.val_node_ids = []
        self.test_node_ids = []
        self.vocab = []

        self.load_dataset(dataset_path, load_file)

    def get_subgraph(self, node_ids):
        fullgraph = self.full_graph.to_networkx()

        node_to_remove = [n for n in fullgraph.nodes if n not in node_ids]
        subgraph = fullgraph.subgraph(node_ids).copy()

        for node in node_to_remove:
            parents = set()
            children = set()
            pred = deque(fullgraph.predecessors(node)) #predecessors
            succ = deque(fullgraph.successors(node)) #successors
            while pred:
                p_node = pred.popleft()
                if p_node in subgraph:
                    parents.add(p_node)
                else:
                    pred += list(fullgraph.predecessors(p_node))
            while succ:
                c_node = succ.popleft()
                if c_node in subgraph:
                    children.add(c_node)
                else:
                    succ += list(fullgraph.successors(c_node))
            for p_node, c_node in product(parents, children):
                subgraph.add_edge(p_node, c_node)

        return subgraph


    def get_node_list(self, file_path):
        node_list = []
        with open(file_path, "r") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    node_list.append(line)
        return node_list


    def load_dataset(self, dataset_path, load_file):
        if load_file:

            node_file = os.path.join(dataset_path, f"{self.taxonomy_name}.terms")
            edge_file = os.path.join(dataset_path, f"{self.taxonomy_name}.relations")
            embedding_file = os.path.join(dataset_path, f"{self.taxonomy_name}.terms.embedding")
            output_pickle_file = os.path.join(dataset_path, f"{self.taxonomy_name}.pickle.bin")

            if self.partitioned:
                train_node_file = os.path.join(dataset_path, f"{self.taxonomy_name}.terms.train")
                val_node_file = os.path.join(dataset_path, f"{self.taxonomy_name}.terms.val")
                test_node_file = os.path.join(dataset_path, f"{self.taxonomy_name}.terms.test")

            taxonomy = nx.DiGraph()
            taxo_id2taxo = {}

            # load nodes (terms) from node file
            with open(node_file, "r") as fin:
                for line in tqdm(fin, desc="Terms"):
                    line = line.strip()
                    if line:
                        seg = line.split("\t")
                        assert len(seg) == 2, f"Wrong number of segments {line}"
                        taxo = Taxo(taxo_id=seg[0], name=seg[1], display_name=seg[1])
                        taxo_id2taxo[seg[0]] = taxo
                        taxonomy.add_node(taxo)

            # load edges (relations) from edge file
            with open(edge_file, "r") as fin:
                for line in tqdm(fin, desc="Relations"):
                    line = line.strip()
                    if line:
                        seg = line.split("\t")
                        assert len(seg) == 2, f"Wrong number of segments {line}"
                        parent_taxo = taxo_id2taxo[seg[0]]
                        child_taxo = taxo_id2taxo[seg[1]]
                        taxonomy.add_edge(parent_taxo, child_taxo)

            # load embedding features
            embeddings = KeyedVectors.load_word2vec_format(embedding_file)
            print(f"Embedding of size {embeddings.vectors.shape} loaded")


            # generate vocab
            ## taxo_id is the old taxo_id read from {self.name}.terms file, node_id is the new taxon_id from 0 to len(vocab)
            taxo_id2node_id = {node.taxo_id:idx for idx, node in enumerate(taxonomy.nodes())}
            node_id2taxo_id = {v:k for k, v in taxo_id2node_id.items()}
            self.vocab = [taxo_id2taxo[node_id2taxo_id[node_id]].name + "@" + str(node_id) for node_id in node_id2taxo_id]

            ## nodes
            node_features = np.zeros(embeddings.vectors.shape)
            for node_id, taxo_id in node_id2taxo_id.items():
                node_features[node_id, :] = embeddings[taxo_id]
            node_features = torch.FloatTensor(node_features)

            ## edges
            edges = []
            for edge in taxonomy.edges():
                parent_node_id = taxo_id2node_id[edge[0].taxo_id]
                child_node_id = taxo_id2node_id[edge[1].taxo_id]
                edges.append([parent_node_id, child_node_id])

            # generate DGLGraph
            self.full_graph.add_nodes(len(node_id2taxo_id), {'x': node_features})
            self.full_graph.add_edges([e[0] for e in edges], [e[1] for e in edges])

            max_val_size = 1000
            max_test_size = 1000
            # generate validation/test node_indices using existing partitions
            if self.partitioned:
                partitioned_train_node_list = self.get_node_list(train_node_file)
                partitioned_val_node_list = self.get_node_list(val_node_file)
                partitioned_test_node_list = self.get_node_list(test_node_file)

                self.train_node_ids = [taxo_id2node_id[taxo_id] for taxo_id in partitioned_train_node_list]
                self.val_node_ids = [taxo_id2node_id[taxo_id] for taxo_id in partitioned_val_node_list]
                self.test_node_ids = [taxo_id2node_id[taxo_id] for taxo_id in partitioned_test_node_list]

            else:
                # for internal nodes
                #if self.sample_position == 'internal':
                root_node = [node for node in taxonomy.nodes() if taxonomy.in_degree(node) == 0]
                sampled_node_ids = [taxo_id2node_id[node.taxo_id] for node in taxonomy.nodes() if node not in root_node]
                random.seed(47)
                random.shuffle(sampled_node_ids)

                # validation/test size
                val_size = min(int(len(sampled_node_ids) * 0.1), max_val_size)
                test_size = min(int(len(sampled_node_ids) * 0.1), max_test_size)

                self.val_node_ids = sampled_node_ids[:val_size]
                self.test_node_ids = sampled_node_ids[val_size:(val_size + test_size)]
                self.train_node_ids = [node_id for node_id in node_id2taxo_id if node_id not in self.val_node_ids and node_id not in self.test_node_ids]
                # for leaf nodes
                #elif self.sample_position == 'leaf':
                #    leaf_node_ids = []
                #    for node in taxonomy.nodes():
                #        if taxonomy.out_degree(node) == 0: # leaf
                #            leaf_node_ids.append(taxo_id2node_id[node.taxo_id])
                #    random.seed(47)
                #    random.shuffle(leaf_node_ids)

                #    val_size = min(int(len(leaf_node_ids) * 0.1), max_val_size)
                #    test_size = min(int(len(leaf_node_ids) * 0.1), max_test_size)
                #    self.val_node_ids = leaf_node_ids[:val_size]
                #    self.test_node_ids = leaf_node_ids[val_size:(val_size + test_size)]
                #    self.train_node_ids = [node_id for node_id in node_id2taxo_id if node_id not in self.val_node_ids and node_id not in self.test_node_ids]
                #else:
                #    raise ValueError('invalid sample position. sample position should be either internal or leaf.')

                # save to pickle for faster loading next time
                with open(output_pickle_file, 'wb') as fout:
                    data = {
                        "taxonomy_name": self.taxonomy_name,
                        "full_graph": self.full_graph,
                        "train_node_ids": self.train_node_ids,
                        "val_node_ids": self.val_node_ids,
                        "test_node_ids": self.test_node_ids,
                        "vocab": self.vocab
                    }
                    pickle.dump(data, fout, pickle.HIGHEST_PROTOCOL)
                print(f"Save dataset to {output_pickle_file}")

        else: # pickle file loading
            with open(dataset_path, "rb") as fin:
                data = pickle.load(fin)

            self.taxonomy_name = data["taxonomy_name"]
            self.full_graph = data["full_graph"]
            self.train_node_ids = data["train_node_ids"]
            self.val_node_ids = data["val_node_ids"]
            self.test_node_ids = data["test_node_ids"]
            self.vocab = data["vocab"]


class RawDataset(Dataset):
    def __init__(self, graph_dataset, mode="train", sampling=1, max_pos_size=10, neg_size=3, num_neighbors=4, normalize_embed=False, cache_refresh_time=16):
        start_time = time.time()
        self.mode = mode
        self.sampling = sampling

        self.max_pos_size = max_pos_size
        self.neg_size = neg_size
        self.num_neighbors = num_neighbors
        # self.shuffle = shuffle
        # self.drop_last = drop_last
        self.normalize_embed = normalize_embed
        self.cache_refresh_time = cache_refresh_time
        self.node_features = graph_dataset.full_graph.ndata['x']

        fullgraph = graph_dataset.full_graph.to_networkx()
        root_node = [node for node in fullgraph.nodes() if fullgraph.in_degree(node) == 0]

        train_node_ids = graph_dataset.train_node_ids
        #test_node_ids = graph_dataset.test_node_ids

        if len(root_node) > 1:
            self.root = len(fullgraph.nodes)
            for r in root_node:
                fullgraph.add_edge(self.root, r)
            root_vector = torch.mean(self.node_features[root_node], dim=0, keepdim=True)
            self.node_features = torch.cat((self.node_features, root_vector), 0)
            self.vocab = graph_dataset.vocab + ['root', 'leaf']
            train_node_ids.append(self.root)
        else:
            self.root = root_node[0]
            self.vocab = graph_dataset.vocab + ['leaf']
        self.fullgraph = fullgraph

        if mode == 'train':
            self.core_graph = self.get_holdout_subgraph(train_node_ids)
            self.pseudo_leaf_node = len(fullgraph.nodes) # add pseudo leaf node to core graph
            for node in list(self.core_graph.nodes()):
                self.core_graph.add_edge(node, self.pseudo_leaf_node) # add edge to pseudo leaf node

            self.leaf_nodes = [node for node in self.core_graph.nodes() if self.core_graph.out_degree(node) == 1] # core graph leaf nodes
            leaf_vector = torch.zeros((1, self.node_features.size(1)))
            self.node_features = torch.cat((self.node_features, leaf_vector), 0)
            if self.normalize_embed:
                self.node_features = F.normalize(self.node_features, p=2, dim=1)

            interested_node_set = set(train_node_ids) - set([self.root])
            self.node_list = list(interested_node_set)

            self.node2parents, self.node2children, self.node2nbs, self.node2pos, self.node2edge = {}, {}, {self.pseudo_leaf_node: []}, {}, {}
            # among interested node set...
            for node in interested_node_set:
                parents = set(self.core_graph.predecessors(node))
                children = set(self.core_graph.successors(node))
                # remove pseudo leaf node
                if len(children) > 1:
                    children = [i for i in children if i != self.pseudo_leaf_node]
                node_pos = [(pred, succ) for pred in parents for succ in children if pred != succ]

                self.node2parents[node] = parents
                self.node2children[node] = children
                self.node2nbs[node] = parents.union(children)
                self.node2pos[node] = node_pos
                self.node2edge[node] = set(self.core_graph.in_edges(node)).union(set(self.core_graph.out_edges(node)))
            self.node2nbs[self.root] = set([n for n in self.core_graph.successors(self.root) if n != self.pseudo_leaf_node])

            self.val_node_list = graph_dataset.val_node_ids
            subgraph = self.get_holdout_subgraph(graph_dataset.train_node_ids + graph_dataset.val_node_ids)
            self.val_node2pos = self.find_position(graph_dataset.val_node_ids, subgraph)

            self.test_node_list = graph_dataset.test_node_ids
            subgraph = self.get_holdout_subgraph(graph_dataset.train_node_ids + graph_dataset.test_node_ids)
            self.test_node2pos = self.find_position(graph_dataset.test_node_ids, subgraph)

            # negative sampling during train/validation
            self.pointer = 0
            self.all_edges = list(self.get_candidate_positions(self.core_graph))
            self.edge2dist = {(u, v): nx.shortest_path_length(self.core_graph, u, v) for (u, v) in self.all_edges}
            random.shuffle(self.all_edges)


        elif mode == 'test':
            self.core_graph = self.fullgraph
            self.pseudo_leaf_node = len(fullgraph.nodes)
            self.node_list = list(self.core_graph.nodes())
            for node in self.node_list:
                self.core_graph.add_edge(node, self.pseudo_leaf_node)
            self.leaf_nodes = [node for node in self.core_graph.nodes() if self.core_graph.out_degree(node) == 1]

            leaf_vector = torch.zeros((1, self.node_features.size(1)))
            self.node_features = torch.cat((self.node_features, leaf_vector), 0)

            if self.normalize_embed:
                self.node_features = F.normalize(self.node_features, p=2, dim=1)

            self.node2parents, self.node2children, self.node2nbs, self.node2pos, self.node2edge = {}, {}, {self.pseudo_leaf_node: []}, {}, {}
            # among node list...
            for node in self.node_list:
                parents = set(self.core_graph.predecessors(node))
                children = set(self.core_graph.successors(node))
                node_pos = [(pred, succ) for pred in parents for succ in children if pred != succ]

                self.node2parents[node] = parents
                self.node2children[node] = children
                self.node2nbs[node] = parents.union(children)
                self.node2pos[node] = node_pos
                self.node2edge[node] = set(self.core_graph.in_edges(node)).union(set(self.core_graph.out_edges(node)))

            self.all_edges = list(self.get_candidate_positions(self.core_graph))

        # else:
        #     raise ValueError('invalid mode. mode should be either train or test.')

        end_time = time.time()
        print(f"Finish loading dataset ({end_time - start_time} seconds)")

    def __getitem__(self, idx):
        # generate data instance: list of (anchor egonet, query node feature, label) triplets
        query = self.node_list[idx]
        res = []

        # generate positive triplets
        if self.sampling == 0:
            pos_positions = self.node2pos[query]
            if len(pos_positions) > self.max_pos_size and self.mode == 'train':
                pos_positions = random.sample(pos_positions, k=self.max_pos_size)
            for u, v in pos_positions:
                res.append([u, v, query, (1, 1, 1, 1)])

        elif self.sampling > 0:
            u, v = random.choice(self.node2pos[query])
            res.append([u, v, query, (1, 1, 1, 1)])

        # select negative parents
        neg_size = len(res) if self.neg_size == -1 else self.neg_size
        neg_anchors = self.get_negative_anchors(query, neg_size)

        # generate negative triplets
        for u, v in neg_anchors:
            u_flag = int(u in self.node2parents[query])
            v_flag = int(v in self.node2children[query])
            e_flag = int(self.edge2dist[(u, v)] <= 2)

            res.append([u, v, query, (0, u_flag, v_flag, e_flag)])

        return tuple(res)

    def __len__(self):
        return len(self.node_list)

    def get_candidate_positions(self, g):
        node2desc = {n: set(descendants(g, n)) for n in g.nodes}
        candidates = set(chain.from_iterable([[(n, d) for d in desc] for n, desc in node2desc.items()]))
        return candidates

    def get_holdout_subgraph(self, node_ids):
        subgraph = self.fullgraph.subgraph(node_ids).copy()
        node_to_remove = [n for n in self.fullgraph.nodes if n not in node_ids]
        for node in node_to_remove:
            parents = set()
            children = set()
            pred = deque(self.fullgraph.predecessors(node))
            succ = deque(self.fullgraph.successors(node))
            while pred:
                p_node = pred.popleft()
                if p_node in subgraph:
                    parents.add(p_node)
                else:
                    pred += list(self.fullgraph.predecessors(p_node))
            while succ:
                c_node = succ.popleft()
                if c_node in subgraph:
                    children.add(c_node)
                else:
                    succ += list(self.fullgraph.successors(c_node))
            for p_node, c_node in product(parents, children):
                subgraph.add_edge(p_node, c_node)

        node2desc = {n: set(descendants(subgraph, n)) for n in subgraph.nodes}
        for node in subgraph.nodes():
            if subgraph.out_degree(node) > 1:
                succ1 = set(subgraph.successors(node))
                succ2 = set(chain.from_iterable([node2desc[n] for n in succ1]))
                succ_set = succ1.intersection(succ2)
                if succ_set:
                    for s in succ_set:
                        subgraph.remove_edge(node, s)

        return subgraph

    def find_position(self, node_ids, subgraph):
        node2pos = {}
        coregraph = self.core_graph
        for node in node_ids:
            parents = set()
            children = set()
            pred = deque(subgraph.predecessors(node)) #predecessors
            succ = deque(subgraph.successors(node)) #successors
            while pred:
                p_node = pred.popleft()
                if p_node in coregraph:
                    parents.add(p_node)
                else:
                    pred += list(subgraph.predecessors(p_node))
            while succ:
                c_node = succ.popleft()
                if c_node in coregraph:
                    children.add(c_node)
                else:
                    succ += list(subgraph.successors(c_node))

            if not children:
                children.add(self.pseudo_leaf_node)
            pos = [(p_node, c_node) for p_node in parents for c_node in children if p_node != c_node]
            node2pos[node] = pos

        return node2pos

    def get_negative_anchors(self, query, neg_size):
        if self.pointer == 0:
            random.shuffle(self.all_edges)

        if self.sampling == 0: # generate at most k(neg_size) negative samples for the query node
            while True:
                negatives = [e for e in self.all_edges[self.pointer: self.pointer + neg_size] if
                             e not in self.node2pos[query] and e not in self.node2edge[query]]
                if len(negatives) > 0:
                    break
                self.pointer += neg_size
                if self.pointer >= len(self.all_edges):
                    self.pointer = 0

        elif self.sampling == 1: # generate exactly k(neg_size) negative samples for the query node
            negatives = []
            while len(negatives) != neg_size:
                num = neg_size - len(negatives)
                negatives.extend([e for e in self.all_edges[self.pointer: self.pointer + num]
                                  if e not in self.node2pos[query] and e not in self.node2edge[query]])
                self.pointer += num
                if self.pointer >= len(self.all_edges):
                    self.pointer = 0
                    random.shuffle(self.all_edges)
            if len(negatives) > neg_size:
                negatives = negatives[:neg_size]

        return negatives

class GraphDataset(RawDataset):
    def __init__(self, graph_dataset, mode="train", sampling=1, max_pos_size=10, neg_size=3, num_neighbors=4, normalize_embed=False, cache_refresh_time=16):
        super(GraphDataset, self).__init__(graph_dataset, mode, sampling, max_pos_size, neg_size, num_neighbors, normalize_embed, cache_refresh_time)

        # local subgraph
        # caching
        self.cache = {}  # if g = self.cache[anchor_node], then g is the egonet centered on the anchor_node
        self.cache_counter = {}  # if n = self.cache[anchor_node], then n is the number of times you used this cache

        local_subgraph = dgl.DGLGraph()
        local_subgraph.add_nodes(1, {"id": torch.tensor([self.pseudo_leaf_node]), "pos": torch.tensor([1])})
        local_subgraph.add_edges(local_subgraph.nodes(), local_subgraph.nodes())
        self.cache[self.pseudo_leaf_node] = local_subgraph

    def __getitem__(self, idx): # geterate data instance: (anchor egonet, query node feature, label)
        res = []
        query = self.node_list[idx]
        # generate positive triplets
        if self.sampling > 0:
            u, v = random.choice(self.node2pos[query])
            u_egonet, v_egonet = self.get_egonet_node_feature(query, u, v)
            res.append([u, v, u_egonet, v_egonet, query, (1, 1, 1, 1)])

        elif self.sampling == 0:
            pos_positions = self.node2pos[query]
            if len(pos_positions) > self.max_pos_size and self.mode == 'train':
                pos_positions = random.sample(pos_positions, k=self.max_pos_size)
            for u, v in pos_positions:
                u_egonet, v_egonet = self.get_egonet_node_feature(query, u, v)
                res.append([u, v, u_egonet, v_egonet, query, (1, 1, 1, 1)])

        # select negative parents
        neg_size = len(res) if self.neg_size == -1 else self.neg_size
        neg_anchors = self.get_negative_anchors(query, neg_size)

        # generate negative triplets
        for u, v in neg_anchors:
            u_egonet, v_egonet = self.get_egonet_node_feature(query, u, v)
            u_flag = int(u in self.node2parents[query])
            v_flag = int(v in self.node2children[query])
            e_flag = int(self.edge2dist[(u, v)] <= 2)
            res.append([u, v, u_egonet, v_egonet, query, (0, u_flag, v_flag, e_flag)])

        return tuple(res)

    def check_cache_flag(self, node):
        return (node in self.cache) and (self.cache_counter[node] < self.cache_refresh_time)

    def get_egonet_node_feature(self, query, anchor_u, anchor_v):
        # for anchor node u
        if anchor_u == self.pseudo_leaf_node:
            u_graph = self.cache[anchor_u]
        else:
            u_flag = ((query < 0) or (anchor_u not in self.node2nbs[query])) and (anchor_u not in self.node2nbs[anchor_v])
            u_cache_flag = self.check_cache_flag(anchor_u)

            if u_flag and u_cache_flag:
                u_graph = self.cache[anchor_u]
                self.cache_counter[anchor_u] += 1
            else:
                u_graph = self.get_subgraph(query, anchor_u, anchor_v, u_flag)
                if u_flag:
                    self.cache[anchor_u] = u_graph
                    self.cache_counter[anchor_u] = 0
        # for anchor node v
        if anchor_v == self.pseudo_leaf_node:
            v_graph = self.cache[anchor_v]
        else:
            v_flag = ((query < 0) or (anchor_v not in self.node2nbs[query])) and (anchor_v not in self.node2nbs[anchor_u])
            v_cache_flag = self.check_cache_flag(anchor_v)

            if v_flag and v_cache_flag:
                v_graph = self.cache[anchor_v]
                self.cache_counter[anchor_v] += 1
            else:
                v_graph = self.get_subgraph(query, anchor_v, anchor_u, v_flag)
                if v_flag:
                    self.cache[anchor_v] = v_graph
                    self.cache_counter[anchor_v] = 0

        return u_graph, v_graph

    def get_subgraph(self, query, anchor, other_anchor, flag):
        if flag:
            if anchor == self.pseudo_leaf_node:
                nodes = [n for n in random.choices(self.leaf_nodes, k = self.num_neighbors)]
                positions = [0] * len(nodes)

                # anchor node itself
                parent_node_idx = len(nodes)
                nodes.append(anchor)
                positions.append(1)

            else:
                nodes = [n for n in self.core_graph.predecessors(anchor)]
                positions = [0] * len(nodes)

                # anchor node itself
                parent_node_idx = len(nodes)
                nodes.append(anchor)
                positions.append(1)

                # sibling os query node (children of anchor node)
                if self.core_graph.out_degree(anchor) <= self.num_neighbors:
                    siblings = [n for n in self.core_graph.successors(anchor) if n != self.pseudo_leaf_node]
                else:
                    siblings = [n for n in random.choices(list(self.core_graph.successors(anchor)), k = self.num_neighbors)
                                if n != self.pseudo_leaf_node]

                nodes.extend(siblings)
                positions.extend([2] * len(siblings))

        else:
            if anchor == self.pseudo_leaf_node:
                nodes = [n for n in random.choices(self.leaf_nodes, k = self.num_neighbors) if n != query and n != other_anchor]
                positions = [0] * len(nodes)

                # anchor node itself
                parent_node_idx = len(nodes)
                nodes.append(anchor)
                positions.append(1)

            else:
                nodes = [n for n in self.core_graph.predecessors(anchor) if n != query and n != other_anchor]
                positions = [0] * len(nodes)

                # anchor node itself
                parent_node_idx = len(nodes)
                nodes.append(anchor)
                positions.append(1)

                # siblings of query node (children of anchor node)
                if self.core_graph.out_degree(anchor) <= self.num_neighbors:
                    siblings = [n for n in self.core_graph.successors(anchor) if
                                n != self.pseudo_leaf_node and n != query and n != other_anchor]
                else:
                    siblings = [n for n in random.choices(list(self.core_graph.successors(anchor)), k = self.num_neighbors)
                                if n != self.pseudo_leaf_node and n != query and n != other_anchor]

                nodes.extend(siblings)
                positions.extend([2] * len(siblings))

        # generate dgl graph with features
        g = dgl.DGLGraph()
        g.add_nodes(len(nodes), {"id": torch.tensor(nodes), "pos": torch.tensor(positions)})

        add_edge_for_dgl(g, list(range(parent_node_idx)), parent_node_idx)
        add_edge_for_dgl(g, parent_node_idx, list(range(parent_node_idx + 1, len(nodes))))

        # add self-cycle
        g.add_edges(g.nodes(), g.nodes())

        return g

def add_edge_for_dgl(g, n1, n2):
    """
    https://github.com/dmlc/dgl/issues/1476 there is a bug in dgl add edges, so we need a wrapper
    """
    if not ((isinstance(n1, list) and len(n1) == 0) or (isinstance(n2, list) and len(n2) == 0)):
        g.add_edges(n1, n2)