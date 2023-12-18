import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, APPNPConv
from GNN.SAGE import SAGE
from GNN.GAT import GAT


class MLP(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        norm_type="none",
    ):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = F.relu(h)
                h = self.dropout(h)
        return h_list, h


class GCN(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        activation,
        norm_type="none",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(GraphConv(input_dim, output_dim, activation=activation))
        else:
            self.layers.append(GraphConv(input_dim, hidden_dim, activation=activation))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(
                    GraphConv(hidden_dim, hidden_dim, activation=activation)
                )
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(GraphConv(hidden_dim, output_dim))

    def forward(self, g, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = self.dropout(h)
        return h_list, h


class APPNP(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        activation,
        norm_type="none",
        edge_drop=0.5,
        alpha=0.1,
        k=10,
    ):

        super(APPNP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.activation = activation
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, output_dim))

        self.propagate = APPNPConv(k, alpha, edge_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, g, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            h = layer(h)

            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = self.activation(h)
                h = self.dropout(h)

        h = self.propagate(g, h)
        return h_list, h


def Model(conf):
    if "MLP" in conf["model_name"]:
        return MLP(
            num_layers=conf["num_layers"],
            input_dim=conf["feat_dim"],
            hidden_dim=conf["hidden_dim"],
            output_dim=conf["prompts_dim"],
            dropout_ratio=conf["dropout_ratio"],
            norm_type=conf["norm_type"],
        ).to(conf["device"])
    elif "SAGE" in conf["model_name"]:
        return SAGE(
            num_layers=conf["num_layers"],
            input_dim=conf["feat_dim"],
            hidden_dim=conf["hidden_dim"],
            output_dim=conf["prompts_dim"],
            dropout_ratio=conf["dropout_ratio"],
            activation=F.relu,
            norm_type=conf["norm_type"],
            input_drop=conf["dropout_ratio"],
        ).to(conf["device"])
    elif "GCN" in conf["model_name"]:
        return GCN(
            num_layers=conf["num_layers"],
            input_dim=conf["feat_dim"],
            hidden_dim=conf["hidden_dim"],
            output_dim=conf["prompts_dim"],
            dropout_ratio=conf["dropout_ratio"],
            activation=F.relu,
            norm_type=conf["norm_type"],
        ).to(conf["device"])
    elif "GAT" in conf["model_name"]:
        return GAT(
            num_layers=conf["num_layers"],
            input_dim=conf["feat_dim"],
            hidden_dim=conf["hidden_dim"],
            output_dim=conf["prompts_dim"],
            dropout_ratio=conf["dropout_ratio"],
            activation=F.relu,
            norm_type=conf["norm_type"],
            input_drop=conf["dropout_ratio"],
            attn_drop=conf["attn_dropout_ratio"],
        ).to(conf["device"])
    elif "APPNP" in conf["model_name"]:
        return APPNP(
            num_layers=conf["num_layers"],
            input_dim=conf["feat_dim"],
            hidden_dim=conf["hidden_dim"],
            output_dim=conf["prompts_dim"],
            dropout_ratio=conf["dropout_ratio"],
            activation=F.relu,
            norm_type=conf["norm_type"],
        ).to(conf["device"])
