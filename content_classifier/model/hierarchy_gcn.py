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

        if self.config.model.structure_encoder.layernorm:
            self.ln_topdown_gate = nn.LayerNorm([len(label_ids), 1])
            self.ln_bottomup_gate = nn.LayerNorm([len(label_ids), 1])
            self.ln_loop_gate = nn.LayerNorm([len(label_ids), 1])
            self.ln_topdown_message = nn.LayerNorm([len(label_ids), self.in_dim])
            self.ln_bottomup_message = nn.LayerNorm([len(label_ids), self.in_dim])
            self.ln_loop_message = nn.LayerNorm([len(label_ids), self.in_dim])

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.config.model.structure_encoder.dropout)
    
    def forward(self, inputs):
        # MESSAGE = (PRIOR * INPUT + BIAS) * GATE
        # GATE = SIGMOID(LINEAR(INPUT + BIAS))
        topdown_gate = self.topdown_fc(inputs + self.topdown_bias2)
        bottomup_gate = self.bottomup_fc(inputs + self.bottomup_bias2)
        loop_gate = self.self.loop_fc(inputs)
        
        if self.config.model.structure_encoder.layernorm:
            topdown_gate = self.ln_topdown_gate(topdown_gate)
            bottomup_gate = self.ln_bottomup_gate(bottomup_gate)
            loop_gate = self.ln_loop_gate(loop_gate)

        topdown_message = (torch.matmul(self.topdown_prior, inputs) + self.topdown_bias1) * F.sigmoid(topdown_gate)
        bottomup_message = (torch.matmul(self.bottomup_prior, inputs) + self.bottomup_bias1) * F.sigmoid(bottomup_gate)
        loop_message = inputs * F.sigmoid(loop_gate)

        if self.config.model.structure_encoder.layernorm:
            topdown_message = self.ln_topdown_message(topdown_message)
            bottomup_message = self.ln_bottomup_message(bottomup_message)
            loop_message = self.ln_loop_message(loop_message)

        return self.relu(self.dropout(topdown_message) + self.dropout(bottomup_message) + self.dropout(loop_message))