"""
GAT implemetation from dgl examples with modifications
https://github.com/dmlc/dgl/blob/master/examples/pytorch/ogb/ogbn-products/gat/models.py
"""

import torch
import torch.nn as nn
from dgl.nn import GATConv


class GAT(nn.Module):
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
        attn_drop,
        num_heads=8,
        negative_slope=0.2,
        residual=False,
    ):
        super().__init__()
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_hidden = num_heads * hidden_dim if i > 0 else input_dim
            out_hidden = hidden_dim

            self.convs.append(
                GATConv(
                    in_hidden,
                    out_hidden,
                    num_heads,
                    0,
                    attn_drop,
                    negative_slope,
                    residual
                )
            )
            if norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(num_heads * out_hidden))
            elif norm_type == "layer":
                self.norms.append(nn.LayerNorm(num_heads * out_hidden))
            else:
                self.norms.append(nn.Identity())

        self.pred_linear = nn.Linear(num_heads * hidden_dim, output_dim)

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout_ratio)
        self.activation = activation
        self.residual = residual

    def forward(self, g, feats, inference=False):
        if not isinstance(g, list):
            subgraphs = [g] * self.num_layers
        else:
            subgraphs = g

        h = self.input_drop(feats)

        h_last = None

        for i in range(self.num_layers):
            h = self.convs[i](subgraphs[i], h).flatten(1, -1)

            if self.residual and h_last is not None:
                h += h_last[: h.shape[0], :]

            h_last = h

            h = self.norms[i](h)
            h = self.activation(h)
            h = self.dropout(h)

            if inference:
                torch.cuda.empty_cache()

        h = self.pred_linear(h)

        return h
