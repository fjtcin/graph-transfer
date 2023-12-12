import argparse
from pathlib import Path
import copy
from utils import set_seed

import torch
import dgl

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE, VGAE, GCNConv

set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--variational', action='store_true')
parser.add_argument('--linear', action='store_true')
parser.add_argument('--dataset', type=str, default='cora',
                    choices=['cora', 'citeseer', 'pubmed'])
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--patience', type=int, default=100)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
])
path = Path('data')
dataset = Planetoid(path, args.dataset, transform=transform, split='public', num_train_per_class=10)
data = dataset[0]
split_edges_transform = T.RandomLinkSplit(num_val=0, num_test=0.1, is_undirected=True,
                        split_labels=True, add_negative_train_samples=False)
train_data, _, test_data = split_edges_transform(data)


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class VariationalLinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_mu = GCNConv(in_channels, out_channels)
        self.conv_logstd = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


in_channels, out_channels = data.num_features, 128

if not args.variational and not args.linear:
    model = GAE(GCNEncoder(in_channels, out_channels))
elif not args.variational and args.linear:
    model = GAE(LinearEncoder(in_channels, out_channels))
elif args.variational and not args.linear:
    model = VGAE(VariationalGCNEncoder(in_channels, out_channels))
elif args.variational and args.linear:
    model = VGAE(VariationalLinearEncoder(in_channels, out_channels))

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    loss = model.recon_loss(z, train_data.pos_edge_label_index)
    if args.variational:
        loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)

def test(data):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)


best_epoch, best_score_val, count = 0, [0, 0], 0
for epoch in range(1, args.epochs + 1):
    loss = train()
    auc, ap = test(test_data)
    score_val = [auc, ap]
    if all(x > y for x, y in zip(score_val, best_score_val)):
        best_epoch = epoch
        best_score_val = score_val
        state = copy.deepcopy(model.state_dict())
        count = 0
    else:
        count += 1
    if count == args.patience:
        break
print(f'Best valid model at epoch: {best_epoch}, AUC: {best_score_val[0]:.4f}, AP: {best_score_val[1]:.4f}')


model.load_state_dict(state)
model.eval()
with torch.no_grad():
    z = model.encode(data.x, data.edge_index)

dgl_graph = dgl.graph((data.edge_index[0], data.edge_index[1]))
dgl_graph.ndata['feat'] = z
dgl_graph.ndata['label'] = data.y
dgl_graph.ndata['train_mask'] = data.train_mask
dgl_graph.ndata['val_mask'] = data.val_mask
dgl_graph.ndata['test_mask'] = data.test_mask
dgl.save_graphs(f'data/{args.dataset}.bin', [dgl_graph])
