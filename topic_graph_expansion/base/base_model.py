from abc import abstractmethod
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.embedding = None
        self.device = None

    def set_embedding(self, vocab_size, embed_dim, pretrained_embedding=None):
        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=True)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)

    def set_device(self, device):
        self.device = device

    @abstractmethod
    def forward(self, *input):
        raise NotImplementedError