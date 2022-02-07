import torch
from torch.nn.functional import sigmoid


class FocalLoss(torch.nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.gamma = 2.
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')
    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        weight = torch.pow(1 - sigmoid(logits), self.gamma) * targets + torch.pow(sigmoid(logits), self.gamma) * (1 - targets)
        return torch.mean(bce * weight) / (0.5 ** self.gamma)


class RecursiveRegularizationLoss(torch.nn.Module):
    def __init__(self, config, hierarchy):
        super(RecursiveRegularizationLoss, self).__init__()
        if config.training.focal_loss:
            self.fn = FocalLoss()
        else:
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


class LabelContradictionPenalty(torch.nn.Module):
    def __init__(self, config, hierarchy):
        super(LabelContradictionPenalty, self).__init__()
        self.penalty_weight = config.training.label_contradiction_penalty.weight
        self.is_absolute = config.training.label_contradiction_penalty.absolute
        self.margin = config.training.label_contradiction_penalty.margin
        self.hierarchy = hierarchy
    
    def forward(self, logits):
        contradictions = 0.
        for parent in self.hierarchy.keys():
            children = self.hierarchy[parent]
            contradiction = sigmoid(logits[:, parent]) - sigmoid(torch.max(logits[:, children], dim=1)[0])
            if self.is_absolute:
                contradiction = torch.abs(contradiction)
            if self.margin > 0:
                contradiction = contradiction[torch.logical_or(contradiction > self.margin, contradiction < - self.margin)]
            if contradiction.size(0) > 0:
                contradictions = contradictions + torch.sum(contradiction)
        return self.penalty_weight * contradictions / logits.size(0)
