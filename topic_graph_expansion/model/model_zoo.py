import dgl
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=1, feat_drop=0.5, attn_drop=0.5, leaky_relu_alpha=0.2, residual=False):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.fc = nn.Linear(in_dim, num_heads * out_dim, bias=False)
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x
        self.attn_l = nn.Parameter(torch.Tensor(size=(1, num_heads, out_dim)))
        self.attn_r = nn.Parameter(torch.Tensor(size=(1, num_heads, out_dim)))
        nn.init.xavier_normal_(self.fc.weight.data, gain=1.414)
        nn.init.xavier_normal_(self.attn_l.data, gain=1.414)
        nn.init.xavier_normal_(self.attn_r.data, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(leaky_relu_alpha)
        self.softmax = edge_softmax
        self.residual = residual
        if residual:
            if in_dim != out_dim:
                self.res_fc = nn.Linear(in_dim, num_heads * out_dim, bias=False)
                nn.init.xavier_normal_(self.res_fc.weight.data, gain=1.414)
            else:
                self.res_fc = None

    def forward(self, g, feature):
        h = self.feat_drop(feature)
        ft = self.fc(h).reshape((h.shape[0], self.num_heads, -1))
        a1 = (ft * self.attn_l).sum(dim=-1).unsqueeze(-1)
        a2 = (ft * self.attn_r).sum(dim=-1).unsqueeze(-1)

        g.ndata['ft'] = ft
        g.ndata['a1'] = a1
        g.ndata['a2'] = a2

        g.apply_edges(self.edge_attention)
        self.edge_softmax(g)
        g.update_all(fn.src_mul_edge('ft', 'a_drop', 'ft'), fn.sum('ft', 'ft'))
        ret = g.ndata['ft']
        if self.residual:
            if self.res_fc is not None:
                res = self.res_fc(h).reshape((h.shape[0], self.num_heads, -1))
            else:
                res = torch.unsqueeze(h, 1)
            ret = res + ret
        return ret

    def edge_attention(self, edges):
        a = self.leaky_relu(edges.src['a1'] + edges.dst['a2'])
        return {'a': a}

    def edge_softmax(self, g):
        attention = self.softmax(g, g.edata.pop('a'))
        g.edata['a_drop'] = self.attn_drop(attention)


class PGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, pos_dim, num_layers, heads, activation, feat_drop=0.5, attn_drop=0.5, leaky_relu_alpha=0.2, position_vocab_size = 3, residual=False):
        super(PGAT, self).__init__()
        self.num_layers = num_layers
        self.activation = activation
        self.gat_layers = nn.ModuleList()
        self.prop_position_embeddings = nn.ModuleList()

        self.gat_layers.append(GATLayer(in_dim + pos_dim, hidden_dim, heads[0], feat_drop, attn_drop, leaky_relu_alpha, residual))
        self.prop_position_embeddings.append(nn.Embedding(position_vocab_size, pos_dim))

        for l in range(1, num_layers):
            self.gat_layers.append(
                GATLayer(hidden_dim * heads[l - 1] + pos_dim, hidden_dim, heads[l], feat_drop, attn_drop,
                         leaky_relu_alpha, residual))
            self.prop_position_embeddings.append(nn.Embedding(position_vocab_size, pos_dim))

        self.gat_layers.append(GATLayer(hidden_dim * heads[-2] + pos_dim, out_dim, heads[-1], feat_drop, attn_drop, leaky_relu_alpha, residual))
        self.prop_position_embeddings.append(nn.Embedding(position_vocab_size, pos_dim))

    def forward(self, g, features):
        h = features
        positions = g.ndata.pop('pos').to(h.device)
        for l in range(self.num_layers):
            p = self.prop_position_embeddings[l](positions)
            h = self.gat_layers[l](g, torch.cat((h, p), 1)).flatten(1)
            h = self.activation(h)
        p = self.prop_position_embeddings[-1](positions)
        h = self.gat_layers[-1](g, torch.cat((h, p), 1)).mean(1)

        return h

class WMR(nn.Module):
    def __init__(self, position_vocab_size=3):
        super(WMR, self).__init__()
        self.position_weights = nn.Embedding(position_vocab_size, 1)
        self.nonlinear = F.softplus

    def forward(self, g, pos):
        g.ndata['a'] = self.nonlinear(self.position_weights(pos))
        return dgl.mean_nodes(g, 'h', 'a')

class NTN(nn.Module):
    def __init__(self, l_dim, r_dim, k=5, non_linear=F.tanh):
        super(NTN, self).__init__()
        self.u = nn.Linear(k, 1, bias=False)
        self.sigma = non_linear
        self.W = nn.Bilinear(l_dim*2, r_dim, k, bias=False)
        self.V = nn.Linear(l_dim*2 + r_dim, k, bias=False)

    def forward(self, e1, e2, q):
        e = torch.cat((e1, e2), -1)

        return self.u(self.sigma(self.W(e, q) + self.V(torch.cat((e, q), 1))))

class TMN(nn.Module):
    def __init__(self, l_dim, r_dim, k=5, non_linear=nn.LeakyReLU(0.2)):
        super(TMN, self).__init__()
        self.u = nn.Linear(k*3, 1, bias=False)
        self.u1 = nn.Linear(k, 1, bias=False)
        self.u2 = nn.Linear(k, 1, bias=False)
        self.u3 = nn.Linear(k, 1, bias=False)
        self.sigma = non_linear
        self.W1 = nn.Bilinear(l_dim, r_dim, k, bias=True)
        self.W2 = nn.Bilinear(l_dim, r_dim, k, bias=True)
        self.W = nn.Bilinear(l_dim*2, r_dim, k, bias=True)
        self.V1 = nn.Linear(l_dim + r_dim, k, bias=False)
        self.V2 = nn.Linear(l_dim + r_dim, k, bias=False)
        self.V = nn.Linear(l_dim*2 + r_dim, k, bias=False)

        self.control = nn.Sequential(nn.Linear(l_dim*2+r_dim, l_dim*2, bias=False), nn.Sigmoid())
        self.control1 = nn.Sequential(nn.Linear(l_dim+r_dim, l_dim, bias=False), nn.Sigmoid())
        self.control2 = nn.Sequential(nn.Linear(l_dim+r_dim, l_dim, bias=False), nn.Sigmoid())

    def forward(self, n1, n2, q):
        nc1 = n1 * self.control1(torch.cat((n1, q), -1))
        nc2 = n2 * self.control2(torch.cat((n2, q), -1))
        n = torch.cat((n1, n2), 1)
        nc = n * self.control(torch.cat((n, q), -1))

        t1 = self.W1(nc1, q) + self.V1(torch.cat((nc1, q), 1))
        t2 = self.W2(nc2, q) + self.V2(torch.cat((nc2, q), 1))
        t = self.W(nc, q) + self.V(torch.cat((nc, q), 1))
        score1 = self.u1(self.sigma(t1))
        score2 = self.u2(self.sigma(t2))
        score3 = self.u3(self.sigma(t))
        score = self.u(self.sigma(torch.cat((t.detach(), t1.detach(), t2.detach()), -1)))

        if self.training:
            return score, score1, score2, score3
        else:
            return score





