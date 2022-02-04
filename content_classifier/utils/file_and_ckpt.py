import json

import numpy as np
import torch

import utils.logger as logger


def read_hierarchy(config, label_ids):
    # READ HIERARCHY FILE AND MAKE DICTIONARY OF PARENT: [CHILD 1, CHILD 2, ...]
    hierarchy = dict()
    with open(config.path.data.hierarchy, 'r', encoding='utf8') as f:
            for line in f.readlines():
                labels = line.strip().split('\t')
                parent, children = labels[0], labels[1:]
                if parent != 'Root':
                    hierarchy[label_ids[parent]] = [label_ids[child] for child in children]
    return hierarchy


def recursive_sequence(hierarchy, label_id):
    if label_id not in hierarchy.keys():
        return [[label_id]]
    else:
        paths = []
        for child in hierarchy[label_id]:
            paths.extend(recursive_sequence(hierarchy, child))
        for path in paths:
            path.append(label_id)
        return paths

def make_label_sequences(hierarchy, label_ids):
    # MAKE LIST OF TOPICS THAT LEAD TO SUS
    # [[SU 1, TOPIC 1, TOPIC 2, ..., HIGHEST TOPIC], [...], ...]
    flags = [False for _ in label_ids]
    paths = []
    for i in range(len(label_ids)):
        if not flags[i]:
            included_paths = recursive_sequence(hierarchy, i)
            for path in included_paths:
                for topic in path:
                    flags[topic] = True
            paths.extend(included_paths)
    return paths

def make_label_indices(config):
    # READ HIERARCHY FILE AND MAKE (AND WRITE) DICTIONARY LABEL NAME: LABEL ID
    label_ids = dict()
    with open(config.path.data.hierarchy, 'r', encoding='utf8') as f:
        for line in f.readlines():
            labels = line.strip().split('\t')
            for label in labels:
                if label  != 'Root' and label not in label_ids.keys():
                    label_ids[label] = len(label_ids)
    with open(config.path.data.labels, 'w') as json_f:
        json.dump(label_ids, json_f)
    return label_ids


def read_prior(config, label_ids):
    # READ PRIOR WHICH IS A DICTIONARY OF {PARENT: {CHILD 1: PRIOR 1, CHILD 2: PRIOR 2, ...}}
    with open(config.path.data.prior, 'r', encoding='utf8') as f:
        priors = json.load(f)
    top_down_prior = np.zeros((len(label_ids), len(label_ids)))
    bottom_up_prior = np.zeros((len(label_ids), len(label_ids)))
    for parent in priors.keys():
        if parent != 'Root':
            children = priors[parent].keys()
            for child in children:
                top_down_prior[label_ids[parent], label_ids[child]] = priors[parent][child]
                bottom_up_prior[label_ids[child], label_ids[parent]] = 1.
    return top_down_prior, bottom_up_prior


def load_checkpoint(checkpoint, model, optimizer, mode='train'):
    # LOAD MODEL AND OPTIMIZER PARAMETERS FROM CHECKPOINT
    checkpoint = torch.load(checkpoint)
    if isinstance(model, torch.nn.parallel.DataParallel):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    if mode == 'train':
        epoch = checkpoint['epoch'] + 1
        performance = checkpoint['performance']
        optimizer.load_state_dict(checkpoint['optimizer'])
        return epoch, performance
    else:
        epoch = checkpoint['epoch']
        return epoch

def save_checkpoint(checkpoint, epoch, performance, model, optimizer):
    # SAVE MODEL AND OPTIMIZER PARAMETERS TO CHECKPOINT
    checkpoint_dict = dict()
    checkpoint_dict['epoch'] = epoch
    checkpoint_dict['performance'] = performance
    if isinstance(model, torch.nn.parallel.DataParallel):
        checkpoint_dict['state_dict'] = model.module.state_dict()
    else:
        checkpoint_dict['state_dict'] = model.state_dict()
    checkpoint_dict['optimizer'] = optimizer.state_dict()
    torch.save(checkpoint_dict, checkpoint)

