import math

from torch.distributions import RelaxedBernoulli

from torch_sparse import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch.nn as nn
import torch.nn.functional as F
import torch
# from torch_geometric.nn import GCNConv, GATConv, APPNP

from .gnn_modules import APPNP, GIN
from .gcn import GCN
from ...config.util import attenuated_kaiming_uniform_, compute_spectral_topo_loss, \
    scale_factor, contrastive_loss
import torch.nn.init as nn_init

from ...utils import LearnableEdgePerturbation


class GraphLearner(nn.Module):
    def __init__(self, input_size, num_pers=16):
        super(GraphLearner, self).__init__()
        self.weight_tensor = nn.Parameter(torch.Tensor(num_pers, input_size))

    def reset_parameters(self):
        nn_init.kaiming_uniform_(self.weight_tensor, a=math.sqrt(5))

    def forward(self, context):
        expand_weight_tensor = self.weight_tensor.unsqueeze(1)
        context_fc = context.unsqueeze(0) * expand_weight_tensor
        context_norm = F.normalize(context_fc, p=2, dim=-1)
        attention = torch.matmul(context_norm, context_norm.transpose(-1, -2)).mean(0)
        mask = (attention > 0).detach().float()
        attention = attention * mask + 0 * (1 - mask)
        return attention


class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5, act='relu'):
        super(MLP, self).__init__()
        self.act = act
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[0:-1]):
            x = lin(x)
            if self.act == 'relu':
                x = F.relu(x)
            else:
                x = F.leaky_relu(x)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


INF = 1e20
VERY_SMALL_NUMBER = 1e-12


class QModel(nn.Module):
    def __init__(self, gnn_nhid, gnn_dropout,
                 d, c,
                 input_dim_drop, input_dim_add,
                 ci_nhid, ci_gnn_dropout,
                 ratio_of_edge_insert,
                 conf):
        super(QModel, self).__init__()
        self.encoder1 = APPNP(d, gnn_nhid, c, gnn_dropout, conf.model['hops'], act=conf.model['act'])
        self.leA_ep = LearnableEdgePerturbation(input_dim_drop=input_dim_drop,
                                                input_dim_add=input_dim_add,
                                                hidden_dim=ci_nhid,
                                                ratio_of_edge_insert=ratio_of_edge_insert,
                                                dropout=ci_gnn_dropout, act=conf.model['act'])

    def reset_parameters(self):
        self.encoder1.reset_parameters()
        self.leA_ep.reset_parameters()

    def forward(self, feats, n_node, edge_index, eigen_vector, edge_weight, encoder2):
        node_features = feats
        # edge attr construct
        edge_attr = torch.pow(eigen_vector[edge_index[0]] - eigen_vector[edge_index[1]], 2)
        edge_attr = torch.cat(
            [edge_attr, torch.concat([feats[edge_index[0]], feats[edge_index[1]]], dim=1)], dim=1)
        ep_out = self.leA_ep(feats, edge_index,
                             edge_weight,
                             edge_attr=edge_attr,
                             eigen_vectors=eigen_vector)
        ci_edge_index = ep_out[1]
        _ci_edge_index, ci_edge_weights = gcn_norm(
            ci_edge_index, None, n_node, False,
            dtype=node_features.dtype)
        ci_row, ci_col = _ci_edge_index
        ci_adj_sparse = SparseTensor(row=ci_col, col=ci_row, value=ci_edge_weights,
                                     sparse_sizes=(n_node, n_node))
        ci_adj = ci_adj_sparse.to_dense()
        train_index = edge_index
        _edge_index, edge_weight = gcn_norm(
            train_index, None, n_node, False,
            dtype=node_features.dtype)
        row, col = _edge_index
        init_adj_sparse = SparseTensor(row=col, col=row, value=edge_weight,
                                       sparse_sizes=(n_node, n_node))
        init_adj = init_adj_sparse.to_dense()

        node_vec_1 = self.encoder1([node_features, init_adj, True])
        node_vec_2 = encoder2([node_features, ci_adj, True])
        output = 0.5 * node_vec_1 + 0.5 * node_vec_2
        # output = node_vec_2
        return output, ep_out


class PModel(nn.Module):
    def __init__(self, mlp_nhid, mlp_dropout, gnn_nhid, gnn_dropout, graph_learn_num_pers, mlp_layers, d, c,
                 conf):
        super(PModel, self).__init__()

        self.encoder1 = MLP(in_channels=d,
                            hidden_channels=mlp_nhid,
                            out_channels=c,
                            num_layers=mlp_layers,
                            dropout=mlp_dropout,
                            act=conf.model['act'])
        self.graph_learner1 = GraphLearner(input_size=d, num_pers=graph_learn_num_pers)

    def reset_parameters(self):
        self.encoder1.reset_parameters()
        self.graph_learner1.reset_parameters()

    def learn_graph(self, graph_learner, node_features):
        raw_adj = graph_learner(node_features)
        adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)
        return raw_adj, adj

    def forward(self, feats, encoder2):
        node_features = feats
        raw_adj, adj = self.learn_graph(self.graph_learner1, node_features)
        node_vec_1 = self.encoder1(node_features).squeeze(1)
        node_vec_2 = encoder2([node_features, adj, True])
        output = 0.5 * node_vec_1 + 0.5 * node_vec_2
        # output = node_vec_2
        return output


class VIDual_Tab(nn.Module):
    def __init__(self,
                 mlp_nhid,
                 mlp_dropout,
                 gnn_nhid,
                 gnn_dropout,
                 mlp_layers, d, c,
                 graph_learn_num_pers,
                 input_dim_drop,
                 input_dim_add,
                 ci_nhid,
                 ci_dropout,
                 ratio_of_edge_insert,
                 conf):
        super(VIDual_Tab, self).__init__()
        self.P_Model = PModel(mlp_nhid, mlp_dropout, gnn_nhid, gnn_dropout, graph_learn_num_pers, mlp_layers, d, c,
                              conf)
        self.Q_Model = QModel(gnn_nhid, gnn_dropout, d, c,
                              input_dim_drop, input_dim_add,
                              ci_nhid, ci_dropout,
                              ratio_of_edge_insert,
                              conf)
        self.encoder1 = APPNP(d, gnn_nhid, c, gnn_dropout, conf.model['hops'], act=conf.model['act'])
        self.reset_parameters()

    def reset_parameters(self):
        self.P_Model.reset_parameters()
        self.Q_Model.reset_parameters()
        self.encoder1.reset_parameters()

    def forward(self, feats, n_node, edge_index,
                eigen_vector, edge_weight
                ):
        q_y, ep_out = self.Q_Model.forward(feats, n_node, edge_index, eigen_vector, edge_weight, self.encoder1)
        p_y = self.P_Model.forward(feats, self.encoder1)
        return p_y, q_y, ep_out


class ELBONCCILoss(nn.Module):
    def __init__(self, binary=False):
        super(ELBONCCILoss, self).__init__()
        self.binary = binary

    def forward(self, labels, train_mask, p_y, q_y, dta, eta,
                edge_index=None,
                edge_weight=None,
                ep_out=None,
                eigen_value=None,
                num_component=None):
        p_y_train = p_y[train_mask]
        p_y_val_tes = p_y[train_mask == 0]
        q_y_train = q_y[train_mask]
        q_y_val_tes = q_y[train_mask == 0]
        if self.binary:
            y_train = labels[train_mask].to(dtype=torch.float)
            loss_p_y_train = F.binary_cross_entropy_with_logits(p_y_train, y_train)
            kl = torch.mean(torch.sigmoid(q_y_val_tes) * F.logsigmoid(q_y_val_tes)) - \
                 torch.mean(torch.sigmoid(q_y_val_tes) * F.logsigmoid(p_y_val_tes))
            loss_q_y_train = F.binary_cross_entropy_with_logits(q_y_train, y_train)
            c_loss = contrastive_loss(q_y_val_tes.unsqueeze(-1), p_y_val_tes.unsqueeze(-1))
        else:
            y_train = labels[train_mask].long()
            loss_p_y_train = F.nll_loss(F.log_softmax(p_y_train, dim=-1), y_train)
            kl = torch.mean(F.softmax(q_y_val_tes, dim=-1) * F.log_softmax(q_y_val_tes, dim=-1)) - \
                 torch.mean(F.softmax(q_y_val_tes, dim=-1) * F.log_softmax(p_y_val_tes, dim=-1))
            loss_q_y_train = F.nll_loss(F.log_softmax(q_y_train, dim=-1), y_train)
            c_loss = contrastive_loss(q_y_val_tes, p_y_val_tes)

        spectral_topo_loss = compute_spectral_topo_loss(edge_index, edge_weight, ep_out, eigen_value, num_component)
        loss = loss_p_y_train + dta * loss_q_y_train + eta * c_loss + kl + spectral_topo_loss
        # loss = loss_p_y_train + dta * loss_q_y_train + eta * c_loss + kl
        return loss
