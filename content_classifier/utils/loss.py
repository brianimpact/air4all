import torch

class RecursiveRegularizationLoss(torch.nn.Module):
    def __init__(self, config, hierarchy):
        super(RecursiveRegularizationLoss, self).__init__()
        self.fn = torch.nn.BCEWithLogitsLoss()
        self.recursive_regularization_penalty = config.training.recursive_regularization_penalty
        self.hierarchy = hierarchy
    
    def forward(self, logits, targets, weights):
        # BCELOSS
        loss = self.fn(logits, targets.to(logits.device))
        if self.recursive_regularization_penalty > 0:
            for parent in self.hierarchy.keys():
                children = self.hierarchy[parent]
                parent_weights = weights[parent].weight
                for child in children:
                    # RECURSIVE REGULARIZATION LOSS
                    loss += (parent_weights - weights[child].weight).pow(2).sum() * self.recursive_regularization_penalty * 0.5
        return loss