from base import BaseModel
from model.model_zoo import *

class GraphModel(nn.Module):
    def __init__(self):
        super(GraphModel, self).__init__()

    def init(self, **options):
        graph_propagation_method = options['graph_propagation_method']
        graph_readout_method = options['graph_readout_method']
        options = options

        if graph_propagation_method == "PGAT":
            self.parent_graph_propagate = PGAT(
                options["in_dim"], options["hidden_dim"], options["out_dim"], options["pos_dim"],
                num_layers=options["num_layers"], heads=options["heads"], activation=F.leaky_relu,
                feat_drop=options["feat_drop"], attn_drop=options["attn_drop"])
            self.child_graph_propagate = PGAT(
                options["in_dim"], options["hidden_dim"], options["out_dim"], options["pos_dim"],
                num_layers=options["num_layers"], heads=options["heads"], activation=F.leaky_relu,
                feat_drop=options["feat_drop"], attn_drop=options["attn_drop"])
        else:
            assert f"Invalid Graph Propagation Method: {self.graph_propagation_method}"

        if graph_readout_method == "WMR":
            self.parent_readout = WMR()
            self.child_readout = WMR()
        else:
            assert f"Invalid Readout Method: {self.graph_readout_method}"

    def parent_graph_encoder(self, g):
        g = g.to(self.device)
        h = self.embedding(g.ndata['id'].to(self.device))
        pos = g.ndata['pos'].to(self.device)
        g.ndata['h'] = self.parent_graph_propagate(g, h)
        h = self.parent_readout(g, pos)
        return h

    def child_graph_encoder(self, g):
        g = g.to(self.device)
        h = self.embedding(g.ndata['id'].to(self.device))
        pos = g.ndata['pos'].to(self.device)
        g.ndata['h'] = self.child_graph_propagate(g, h)
        h = self.child_readout(g, pos)
        return h

    def forward_graph_encoders(self, graph_u, graph_v):
        hgu = self.parent_graph_encoder(graph_u)
        hgv = self.child_graph_encoder(graph_v)
        return hgu, hgv



class MatchModel(BaseModel, GraphModel):
    def __init__(self, input_mode, **options):
        super(MatchModel, self).__init__()
        self.input_mode = input_mode
        self.options = options

        l_dim = 0
        if 't' in self.input_mode:
            l_dim += options["in_dim"]
        if 'g' in self.input_mode:
            l_dim += options["out_dim"]
            GraphModel.init(self, **options)
        self.l_dim = l_dim
        self.r_dim = options["in_dim"]

        if options['matching_method'] == "NTN":
            self.match = NTN(self.l_dim, self.r_dim, options["k"])
        elif options['matching_method'] == "TMN":
            self.match = TMN(self.l_dim, self.r_dim, options["k"])
        else:
            assert f"Invalid Matching Method: {options['matching_method']}"

    def forward_encoders(self, u=None, v=None, graph_u=None, graph_v=None, lens=None):
        ur, vr = [], []
        if graph_u != None and u != None and u.size()[0] != len(graph_u):
            cuda_device_id = u.device.index
            graph_u = graph_u[u.size()[0] * cuda_device_id: u.size()[0] * (cuda_device_id + 1)]
            graph_v = graph_v[v.size()[0] * cuda_device_id: v.size()[0] * (cuda_device_id + 1)]

        if 't' in self.input_mode:
            hu = self.embedding(u.to(self.device))
            hv = self.embedding(v.to(self.device))
            ur.append(hu)
            vr.append(hv)
        if 'g' in self.input_mode:
            graph_u = dgl.batch(graph_u)
            graph_v = dgl.batch(graph_v)
            hgu, hgv = self.forward_graph_encoders(graph_u, graph_v)
            ur.append(hgu)
            vr.append(hgv)

        ur = torch.cat(ur, -1)
        vr = torch.cat(vr, -1)
        return ur, vr

    def forward(self, q, *inputs):
        qf = self.embedding(q.to(self.device))
        ur, vr = self.forward_encoders(*inputs)
        scores = self.match(ur, vr, qf)
        return scores