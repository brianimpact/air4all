import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.file_and_ckpt import read_prior


class HierarchyGCN(nn.Module):
    def __init__(self, config, label_ids):
        super(HierarchyGCN, self).__init__()
        # top_down_prior[parent][child] = P and bottom_up_prior[child][parent] = 1.
        self.config = config
        topdown_prior, bottomup_prior = read_prior(self.config, label_ids)
        self.register_buffer('topdown_prior', torch.tensor(topdown_prior, dtype=torch.float32, requires_grad=False))
        self.register_buffer('bottomup_prior', torch.tensor(bottomup_prior, dtype=torch.float32, requires_grad=False))
        self.in_dim = self.config.model.structure_encoder.dimension
        # TOPDOWN GCN
        self.topdown_bias1 = nn.Parameter(torch.zeros([1, len(label_ids), self.in_dim], dtype=torch.float32))
        self.topdown_bias2 = nn.Parameter(torch.zeros([1, len(label_ids), 1], dtype=torch.float32))
        self.topdown_fc = nn.Linear(self.in_dim, 1, bias=False)
        # BOTTOMUP GCN
        self.bottomup_bias1 = nn.Parameter(torch.zeros([1, len(label_ids), self.in_dim], dtype=torch.float32))
        self.bottomup_bias2 = nn.Parameter(torch.zeros([1, len(label_ids), 1], dtype=torch.float32))
        self.bottomup_fc = nn.Linear(self.in_dim, 1, bias=False)
        # LOOP CONNECTION GCN
        self.loop_fc = nn.Linear(self.in_dim, 1, bias=False)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.config.model.structure_encoder.dropout)
    
    def forward(self, inputs):
        # MESSAGE = (PRIOR * INPUT + BIAS) * GATE
        # GATE = SIGMOID(LINEAR(INPUT + BIAS))
        topdown_message = (torch.matmul(self.topdown_prior, inputs) + self.topdown_bias1) * F.sigmoid(self.topdown_fc(inputs + self.topdown_bias2))
        bottomup_message = (torch.matmul(self.bottomup_prior, inputs) + self.bottomup_bias1) * F.sigmoid(self.bottomup_fc(inputs + self.bottomup_bias2))
        loop_message = inputs * F.sigmoid(self.loop_fc(inputs))

        return self.relu(self.dropout(topdown_message) + self.dropout(bottomup_message) + self.dropout(loop_message))