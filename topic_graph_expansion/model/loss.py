import numpy as np
import torch
import torch.nn.functional as F
from itertools import product
import re

epsilon = 1e-9


def bce_loss(output, target):
    loss = F.binary_cross_entropy_with_logits(output.squeeze(), target.float(), reduction="mean")
    return loss

def cross_entropy_loss(output, target, beta=1.0):
    loss = F.cross_entropy(output, target.long(), reduction="mean")
    return loss

class DistMarginLoss:
    def __init__(self, shortest_path_dist):
        self.shortest_path_dist = torch.FloatTensor(shortest_path_dist)
        self.shortest_path_dist /= self.shortest_path_dist.max()

    def loss(self, output, target, nodes):
        label = target.cpu().numpy()
        mid_ids = [(m.start() // label.itemsize)+1 for m in re.finditer(np.array([1, 0], dtype=label.dtype).tostring(), label.tostring())]
        end_ids = [(m.start() // label.itemsize)+1 for m in re.finditer(np.array([0, 1], dtype=label.dtype).tostring(), label.tostring())]
        end_ids.append(len(label))
        start_ids = [0] + end_ids[:-1]

        pair_ids = []
        for start, mid, end in zip(start_ids, mid_ids, end_ids):
            pair_ids.extend(list(product(range(start, mid), range(mid, end))))
        positive_ids, negative_ids = zip(*pair_ids)
        positive_ids_list = list(positive_ids)
        negative_ids_list = list(negative_ids)
        positive_node_ids = [nodes[i] for i in positive_ids_list]
        negative_node_ids = [nodes[i] for i in negative_ids_list]
        margins = self.shortest_path_dist[positive_node_ids, negative_node_ids].to(target.device)
        output = output.view(-1)
        loss = (-output[positive_ids_list].sigmoid().clamp(min=epsilon) + output[negative_ids_list].sigmoid().clamp(min=epsilon) + margins.clamp(min=epsilon)).clamp(min=0)

        return loss.mean()