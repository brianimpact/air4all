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
        self.dropout = nn.Dropout(self.config.model.feature_aggregation.dropout)
        # INEFFICIENT CLASSIFIER DEFINED AS MODULE LIST (BUT IS QUITE EFFICIENT FOR RECURSIVE REGULARIZATION AS IT REDUCES TIME BY ~ 65%)
        self.classifiers = nn.ModuleList()
        for _ in self.label_ids:
            self.classifiers.append(nn.Linear(len(self.label_ids) * self.config.model.structure_encoder.dimension, 1))
    
    def forward(self, inputs):
        flattened = inputs.view(inputs.size(0), -1) # B*(#CNN)kD(CNN) --> (#CNN)k acts as the number of CNN feature set
        node_inputs = self.dropout(self.linear_transformation(flattened)) # B*D(NODE)|V|
        node_inputs = node_inputs.view(inputs.size(0), len(self.label_ids), -1) # B*|V|*D(NODE)
        labelwise_text_feature = self.structure_encoder(node_inputs) # B*|V|*D(NODE)
        labelwise_text_feature = labelwise_text_feature.view(inputs.size(0), -1)
        logits = torch.cat([classifier(labelwise_text_feature) for classifier in self.classifiers], dim=1)
        return logits
