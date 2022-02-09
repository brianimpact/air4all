import numpy as np
import itertools
import re

def hit_at_k(all_ranks, k):
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    hits = np.sum(rank_positions <= k)
    return 1.0 * hits / len(rank_positions)

def hit_at_1(all_ranks):
    return hit_at_k(all_ranks, 1)

def hit_at_5(all_ranks):
    return hit_at_k(all_ranks, 5)

def hit_at_10(all_ranks):
    return hit_at_k(all_ranks, 10)

def macro_mean_rank(all_ranks):
    macro_mean_rank = np.array([np.array(all_rank).mean() for all_rank in all_ranks]).mean()
    return macro_mean_rank

def micro_mean_rank(all_ranks):
    micro_mean_rank = np.array(list(itertools.chain(*all_ranks))).mean()
    return micro_mean_rank

def scaled_mean_reciprocal_rank(all_ranks):
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    scaled_rank_positions = np.ceil(rank_positions / 10)
    return (1.0 / scaled_rank_positions).mean()

def get_ranks(outputs, targets, flag):
    all_ranks = []
    pred = outputs.cpu().numpy().squeeze()
    label = targets.cpu().numpy()

    end_ids = [(m.start() // label.itemsize)+1 for m in re.finditer(np.array([0, 1], dtype=label.dtype).tostring(), label.tostring())]
    end_ids.append(len(label)+1)
    start_ids = [0] + end_ids[:-1]

    for start, end in zip(start_ids, end_ids):
        distances = pred[start:end]
        labels = label[start:end]
        positive_relations = list(np.where(label==1)[0])
        if flag == 0:
            ranks = list(np.argsort(np.argsort(distances))[positive_relations]+1)
        else:
            ranks = list(np.argsort(np.argsort(-distances))[positive_relations] + 1)
        all_ranks.append(ranks)
    return all_ranks