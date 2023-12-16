import torch
from dataloader import load_data

g, labels, idx_train, idx_val, idx_test = load_data('ogbn-arxiv', './data')
# compute the prototypical feature for each class as a torch tensor
# the prototype should only be computed on the training nodes
feats = g.ndata["feat"]
feat_dim = feats.shape[1]
label_dim = labels.max().item() + 1
prototypes = torch.zeros(label_dim, feat_dim)
for i in range(label_dim):
    prototypes[i] = feats[idx_train][labels[idx_train] == i].mean(dim=0)
torch.save(prototypes, f'data/ogbn-arxiv_prototypes.pt')
