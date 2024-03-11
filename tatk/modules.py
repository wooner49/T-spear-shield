import torch
import torch.nn as nn
import dgl
from memorys import *
from layers import *

class GeneralModel(torch.nn.Module):

    def __init__(self, dim_node, dim_edge, sample_param, memory_param, gnn_param, train_param, combined=False):
        super(GeneralModel, self).__init__()
        self.dim_node = dim_node
        self.dim_node_input = dim_node
        self.dim_edge = dim_edge
        self.sample_param = sample_param
        self.memory_param = memory_param
        if not 'dim_out' in gnn_param:
            gnn_param['dim_out'] = memory_param['dim_out']
        self.gnn_param = gnn_param
        self.dim_out = gnn_param['dim_out']
        self.train_param = train_param
        if memory_param['type'] == 'node':
            if memory_param['memory_update'] == 'gru':
                self.memory_updater = GRUMemeoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], dim_node)
            elif memory_param['memory_update'] == 'rnn':
                self.memory_updater = RNNMemeoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], dim_node)
            elif memory_param['memory_update'] == 'transformer':
                self.memory_updater = TransformerMemoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], train_param)
            else:
                raise NotImplementedError
            self.dim_node_input = memory_param['dim_out']
        self.layers = torch.nn.ModuleDict()
        if gnn_param['arch'] == 'transformer_attention':
            for h in range(sample_param['history']):
                self.layers['l0h' + str(h)] = TransfomerAttentionLayer(self.dim_node_input, dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out'], combined=combined)
            for l in range(1, gnn_param['layer']):
                for h in range(sample_param['history']):
                    self.layers['l' + str(l) + 'h' + str(h)] = TransfomerAttentionLayer(gnn_param['dim_out'], dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out'], combined=False)
        elif gnn_param['arch'] == 'identity':
            self.gnn_param['layer'] = 1
            for h in range(sample_param['history']):
                self.layers['l0h' + str(h)] = IdentityNormLayer(self.dim_node_input)
                if 'time_transform' in gnn_param and gnn_param['time_transform'] == 'JODIE':
                    self.layers['l0h' + str(h) + 't'] = JODIETimeEmbedding(gnn_param['dim_out'])
        else:
            raise NotImplementedError
        self.edge_predictor = EdgePredictor(gnn_param['dim_out'])
        if 'combine' in gnn_param and gnn_param['combine'] == 'rnn':
            self.combiner = torch.nn.RNN(gnn_param['dim_out'], gnn_param['dim_out'])
    
    def forward(self, mfgs, neg_samples=1):
        if self.memory_param['type'] == 'node':
            self.memory_updater(mfgs[0])
        out = list()
        for l in range(self.gnn_param['layer']):
            for h in range(self.sample_param['history']):
                rst = self.layers['l' + str(l) + 'h' + str(h)](mfgs[l][h])
                if 'time_transform' in self.gnn_param and self.gnn_param['time_transform'] == 'JODIE':
                    rst = self.layers['l0h' + str(h) + 't'](rst, mfgs[l][h].srcdata['mem_ts'], mfgs[l][h].srcdata['ts'])
                if l != self.gnn_param['layer'] - 1:
                    mfgs[l + 1][h].srcdata['h'] = rst
                else:
                    out.append(rst)
        if self.sample_param['history'] == 1:
            out = out[0]
        else:
            out = torch.stack(out, dim=0)
            out = self.combiner(out)[0][-1, :, :]
        return self.edge_predictor(out, neg_samples=neg_samples)

    def get_embeddings(self, mfgs):
        if self.memory_param['type'] == 'node':
            self.memory_updater(mfgs[0])
        out = list()
        for l in range(self.gnn_param['layer']):
            for h in range(self.sample_param['history']):
                rst = self.layers['l' + str(l) + 'h' + str(h)](mfgs[l][h])
                if 'time_transform' in self.gnn_param and self.gnn_param['time_transform'] == 'JODIE':
                    rst = self.layers['l0h' + str(h) + 't'](rst, mfgs[l][h].srcdata['mem_ts'], mfgs[l][h].srcdata['ts'])
                if l != self.gnn_param['layer'] - 1:
                    mfgs[l + 1][h].srcdata['h'] = rst
                else:
                    out.append(rst)
        if self.sample_param['history'] == 1:
            out = out[0]
        else:
            out = torch.stack(out, dim=0)
            out = self.combiner(out)[0][-1, :, :]
        return out

    # @torch.no_grad()
    # def get_score_without_update(self, mfgs, neg_samples=1):
    #     out = list()
    #     for l in range(self.gnn_param['layer']):
    #         for h in range(self.sample_param['history']):
    #             rst = self.layers['l' + str(l) + 'h' + str(h)](mfgs[l][h])
    #             if 'time_transform' in self.gnn_param and self.gnn_param['time_transform'] == 'JODIE':
    #                 rst = self.layers['l0h' + str(h) + 't'](rst, mfgs[l][h].srcdata['mem_ts'], mfgs[l][h].srcdata['ts'])
    #             if l != self.gnn_param['layer'] - 1:
    #                 mfgs[l + 1][h].srcdata['h'] = rst
    #             else:
    #                 out.append(rst)
    #     if self.sample_param['history'] == 1:
    #         out = out[0]
    #     else:
    #         out = torch.stack(out, dim=0)
    #         out = self.combiner(out)[0][-1, :, :]
    #     return self.edge_predictor(out, neg_samples=neg_samples)

class NodeClassificationModel(torch.nn.Module):

    def __init__(self, dim_in, dim_hid, num_class):
        super(NodeClassificationModel, self).__init__()
        self.fc1 = torch.nn.Linear(dim_in, dim_hid)
        self.fc2 = torch.nn.Linear(dim_hid, num_class)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x


class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, adj, symmetric=False):
        super(EstimateAdj, self).__init__()
        n = len(adj)
        self.estimated_adj = nn.Parameter(torch.FloatTensor(n, n))
        self._init_estimation(adj)
        self.symmetric = symmetric
        # self.device = device

    def _init_estimation(self, adj):
        with torch.no_grad():
            n = len(adj)
            self.estimated_adj.data.copy_(adj)

    def forward(self):
        return self.estimated_adj

    def normalize(self):

        if self.symmetric:
            adj = (self.estimated_adj + self.estimated_adj.t())/2
        else:
            adj = self.estimated_adj

        normalized_adj = self._normalize(adj + torch.eye(adj.shape[0]).cuda())
        return normalized_adj

    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx