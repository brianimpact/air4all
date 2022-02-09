import torch
from torch import nn

from model import hierarchy_gcn


class TextFeaturePropagation(nn.Module):
    def __init__(self, config, label_ids):
        super(TextFeaturePropagation, self).__init__()
        self.config = config
        self.label_ids = label_ids
        self.structure_encoder = hierarchy_gcn.HierarchyGCN(self.config, self.label_ids)
        # LINEAR TRANSFORMATION LAYER THAT GENERATED NODE INPUTS FOR GCN
        self.linear_transformation = nn.Linear(self.config.model.cnn.pooling_k * self.config.model.cnn.dimension * len(self.config.model.cnn.kernels),
                                               len(self.label_ids) * self.config.model.structure_encoder.dimension)
        if self.config.model.feature_aggregation.layernorm:
            self.layernorm = nn.LayerNorm(len(self.label_ids) * self.config.model.structure_encoder.dimension)
        self.dropout = nn.Dropout(self.config.model.feature_aggregation.dropout)
        if self.config.training.recursive_regularization_penalty > 0:
            # INEFFICIENT CLASSIFIER DEFINED AS MODULE LIST (BUT IS QUITE EFFICIENT FOR RECURSIVE REGULARIZATION AS IT REDUCES TIME BY ~ 65%)
            self.classifiers = nn.ModuleList()
            for _ in self.label_ids:
                self.classifiers.append(nn.Linear(len(self.label_ids) * self.config.model.structure_encoder.dimension, 1))
        else:
            self.classifiers = nn.Linear(len(self.label_ids) * self.config.model.structure_encoder.dimension, len(self.label_ids))
    
    def forward(self, inputs):
        flattened = inputs.view(inputs.size(0), -1) # B*(#CNN)kD(CNN) --> (#CNN)k acts as the number of CNN feature set
        node_inputs = self.linear_transformation(flattened) # B*D(NODE)|V|
        if self.config.model.feature_aggregation.layernorm:
            node_inputs = self.layernorm(node_inputs)
        node_inputs = self.dropout(node_inputs) # B*D(NODE)|V|
        node_inputs = node_inputs.view(inputs.size(0), len(self.label_ids), -1) # B*|V|*D(NODE)
        labelwise_text_feature = self.structure_encoder(node_inputs) # B*|V|*D(NODE)
        labelwise_text_feature = labelwise_text_feature.view(inputs.size(0), -1)
        if isinstance(self.classifiers, nn.ModuleList):
            logits = torch.cat([classifier(labelwise_text_feature) for classifier in self.classifiers], dim=1)
        else:
            logits = self.classifiers(labelwise_text_feature)
        return logits
