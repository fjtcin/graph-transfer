import torch
import torch.nn as nn
from dgl.nn import SAGEConv


class SAGE(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        activation,
        norm_type,
        input_drop,
        aggregator_type='gcn',
    ):
        super().__init__()
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_hidden = hidden_dim if i > 0 else input_dim
            out_hidden = hidden_dim if i < num_layers - 1 else output_dim

            self.convs.append(
                SAGEConv(
                    in_hidden,
                    out_hidden,
                    aggregator_type,
                )
            )
            if i < num_layers - 1:  # No normalization after the last layer
                if norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(out_hidden))
                elif norm_type == "layer":
                    self.norms.append(nn.LayerNorm(out_hidden))
                else:
                    self.norms.append(nn.Identity())

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout_ratio)
        self.activation = activation

    def forward(self, g, feats, inference=False):
        if not isinstance(g, list):
            subgraphs = [g] * self.num_layers
        else:
            subgraphs = g

        h = self.input_drop(feats)

        for i in range(self.num_layers):
            h = self.convs[i](subgraphs[i], h)

            if i < self.num_layers - 1:  # No normalization or activation after the last layer
                h = self.norms[i](h)
                h = self.activation(h)
                h = self.dropout(h)

            if inference:
                torch.cuda.empty_cache()

        return h
