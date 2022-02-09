import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LayerNormGRUCell, self).__init__()
        self.W_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_r = nn.Linear(input_size, hidden_size, bias=False)
        self.W_u = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_u = nn.Linear(input_size, hidden_size, bias=False)
        self.W_c = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_c = nn.Linear(input_size, hidden_size, bias=False)
        
        self.ln_r_hidden = nn.LayerNorm(hidden_size)
        self.ln_r_input = nn.LayerNorm(hidden_size)
        self.ln_u_hidden = nn.LayerNorm(hidden_size)
        self.ln_u_input = nn.LayerNorm(hidden_size)
        self.ln_c_hidden = nn.LayerNorm(hidden_size)
        self.ln_c_input = nn.LayerNorm(hidden_size)

        for ln in [self.ln_r_hidden, self.ln_r_input, self.ln_u_hidden, self.ln_u_input, self.ln_c_hidden, self.ln_c_input]:
            nn.init.constant_(ln.weight, 0.1)
            nn.init.constant_(ln.bias, 0.)

    def forward(self, x, h):
        reset_gate = F.sigmoid(self.ln_r_hidden(self.W_r(h)) + self.ln_r_input(self.U_r(x)))
        update_gate = F.sigmoid(self.ln_u_hidden(self.W_u(h)) + self.ln_u_input(self.U_u(x)))
        candidate = F.tanh(self.ln_c_hidden(self.W_c(h * reset_gate)) + self.ln_c_input(self.U_c(x)))
        next_h = (1. - update_gate) * h + update_gate * candidate
        
        return next_h


class LayerNormGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, batch_first):
        super(LayerNormGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        assert self.num_layers > 0
        self.gru_cells = nn.ModuleDict()
        for layer in range(num_layers):
            if layer == 0:
                layer_input_size = input_size
            else:
                layer_input_size = hidden_size
            self.gru_cells['layer%d_forward' % layer] = LayerNormGRUCell(layer_input_size, hidden_size)
            if bidirectional:
                self.gru_cells['layer%d_backward' % layer] = LayerNormGRUCell(layer_input_size, hidden_size)
    
    def forward(self, embedding, lengths):
        if not self.batch_first:
            embedding = embedding.transpose(0, 1)
        forward_embedding = embedding.copy()
        backward_embedding = embedding.copy()

        for layer in range(self.num_layers):
            hiddens = []
            hidden = torch.zeros((embedding.size(0), self.hidden_size))
            for t in range(embedding.size(1)):
                hidden = self.gru_cells['layer%d_forward' % layer](forward_embedding[:, t, :], hidden)
                hiddens.append(hidden)
            forward_embedding = [hiddens[(lengths[i]).item() - 1][i, :] for i in range(embedding.size(0))]
            forward_embedding = torch.stack(forward_embedding, dim=0)

        if self.bidirectional:
            for layer in range(self.num_layers):
                hidden = torch.zeros((embedding.size(0), self.hidden_size))
                for t in range(embedding.size(1))[::-1]:
                    hidden = self.gru_cells['layer%d_forward' % layer](backward_embedding[:, t, :], hidden)
                    for i in range(embedding.size(0)):
                        if lengths[i].item() <= t:
                            hidden[i, :] = 0.
                backward_embedding = hidden
            return torch.cat([forward_embedding, backward_embedding], dim=1)
        else:
            return forward_embedding