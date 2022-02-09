import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNormGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LayerNormGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.W_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_r = nn.Linear(input_size, hidden_size, bias=False)
        self.W_u = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_u = nn.Linear(input_size, hidden_size, bias=False)
        self.W_c = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_c = nn.Linear(input_size, hidden_size, bias=False)

        self.b_r = nn.Parameter(torch.empty((1, hidden_size)))
        self.b_u = nn.Parameter(torch.empty((1, hidden_size)))
        self.b_c = nn.Parameter(torch.empty((1, hidden_size)))
        
        self.ln_r_hidden = nn.LayerNorm(hidden_size)
        self.ln_r_input = nn.LayerNorm(hidden_size)
        self.ln_u_hidden = nn.LayerNorm(hidden_size)
        self.ln_u_input = nn.LayerNorm(hidden_size)
        self.ln_c_hidden = nn.LayerNorm(hidden_size)
        self.ln_c_input = nn.LayerNorm(hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        for linear in [self.W_r, self.U_r, self.W_u, self.U_u, self.W_c, self.U_c]:
            nn.init.uniform_(linear.weight, -self.hidden_size ** (-0.5), self.hidden_size ** (-0.5))
        
        for bias in [self.b_r, self.b_u, self.b_c]:
            nn.init.uniform_(bias, -self.hidden_size ** (-0.5), self.hidden_size ** (-0.5))

        for ln in [self.ln_r_hidden, self.ln_r_input, self.ln_u_hidden, self.ln_u_input, self.ln_c_hidden, self.ln_c_input]:
            nn.init.constant_(ln.weight, 0.1)
            nn.init.constant_(ln.bias, 0.)

    def forward(self, x, h):
        reset_gate = F.sigmoid(self.ln_r_hidden(self.W_r(h)) + self.ln_r_input(self.U_r(x)) + self.b_r)
        update_gate = F.sigmoid(self.ln_u_hidden(self.W_u(h)) + self.ln_u_input(self.U_u(x)) + self.b_u)
        candidate = F.tanh(self.ln_c_hidden(self.W_c(h * reset_gate)) + self.ln_c_input(self.U_c(x)) + self.b_c)
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
        forward_embedding = embedding.clone()
        backward_embedding = embedding.clone()

        for layer in range(self.num_layers):
            hiddens = []
            hidden = torch.zeros((embedding.size(0), self.hidden_size)).to(embedding.device)
            for t in range(embedding.size(1)):
                hidden = self.gru_cells['layer%d_forward' % layer](forward_embedding[:, t, :], hidden)
                hiddens.append(hidden)
                for i in range(embedding.size(0)):
                    if lengths[i].item() <= t:
                        hidden[i, :] = 0.
            forward_embedding = torch.stack(hiddens, dim=1)

        if self.bidirectional:
            for layer in range(self.num_layers):
                hiddens = []
                hidden = torch.zeros((embedding.size(0), self.hidden_size)).to(embedding.device)
                for t in range(embedding.size(1))[::-1]:
                    hidden = self.gru_cells['layer%d_backward' % layer](backward_embedding[:, t, :], hidden)
                    for i in range(embedding.size(0)):
                        if lengths[i].item() <= t:
                            hidden[i, :] = 0.
                    hiddens.append(hidden)
                backward_embedding = torch.stack(hiddens[::-1], dim=1)
            return torch.cat([forward_embedding, backward_embedding], dim=2)
        else:
            return forward_embedding