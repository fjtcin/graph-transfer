"""
Dataloader of CPF datasets are adapted from the CPF implementation
https://github.com/BUPT-GAMMA/CPF/tree/389c01aaf238689ee7b1e5aba127842341e123b6/data

Dataloader of NonHom datasets are adapted from the Non-homophily benchmarks
https://github.com/CUAI/Non-Homophily-Benchmarks

Dataloader of BGNN datasets are adapted from the BGNN implementation and dgl example of BGNN
https://github.com/nd7141/bgnn
https://github.com/dmlc/dgl/tree/473d5e0a4c4e4735f1c9dc9d783e0374328cca9a/examples/pytorch/bgnn
"""

import numpy as np
import scipy.sparse as sp
import torch
import dgl
import os
import scipy
import pandas as pd
import json
from dgl.data.utils import load_graphs
from category_encoders import CatBoostEncoder
from pathlib import Path
from google_drive_downloader import GoogleDriveDownloader as gdd
from sklearn.preprocessing import label_binarize
from sklearn import preprocessing
from ogb.nodeproppred import DglNodePropPredDataset


CPF_data = ["cora", "citeseer", "pubmed", "a-computer", "a-photo"]
OGB_data = ["ogbn-arxiv", "ogbn-products"]
NonHom_data = ["pokec", "penn94"]
BGNN_data = ["house_class", "vk_class"]


def load_data(dataset, dataset_path, **kwargs):
    if dataset in CPF_data:
        return load_cpf_data(
            dataset,
            dataset_path
        )
    elif dataset in OGB_data:
        return load_ogb_data(dataset, dataset_path)
    elif dataset in NonHom_data:
        return load_nonhom_data(dataset, dataset_path, kwargs["split_idx"])
    elif dataset in BGNN_data:
        return load_bgnn_data(dataset, dataset_path, kwargs["split_idx"])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def load_ogb_data(dataset, dataset_path):
    data = DglNodePropPredDataset(dataset, dataset_path)
    splitted_idx = data.get_idx_split()
    idx_train, idx_val, idx_test = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )

    g, labels = data[0]
    labels = labels.squeeze()

    # Turn the graph to undirected
    if dataset == "ogbn-arxiv":
        srcs, dsts = g.edges()
        g.add_edges(dsts, srcs)
        g = g.add_self_loop().to_simple()

    assert has_all_reversed_edges(g) and has_self_loop_every_node(g) and is_not_multigraph(g), 'failed graph check'
    return g, labels, idx_train, idx_val, idx_test


def load_cpf_data(dataset, dataset_path):
    dataset = dgl.data.CitationGraphDataset(dataset, dataset_path, verbose=False)
    g = dataset[0]
    assert has_all_reversed_edges(g) and has_self_loop_every_node(g) and is_not_multigraph(g), 'failed graph check'
    return g, g.ndata['label'], np.where(g.ndata['train_mask'])[0], np.where(g.ndata['val_mask'])[0], np.where(g.ndata['test_mask'])[0]


def load_nonhom_data(dataset, dataset_path, split_idx):
    data_path = Path.cwd().joinpath(dataset_path, f"{dataset}.mat")
    data_split_path = Path.cwd().joinpath(
        dataset_path, "splits", f"{dataset}-splits.npy"
    )

    if dataset == "pokec":
        g, features, labels = load_pokec_mat(data_path)
    elif dataset == "penn94":
        g, features, labels = load_penn94_mat(data_path)
    else:
        raise ValueError("Invalid dataname")

    g = g.remove_self_loop().add_self_loop()
    g.ndata["feat"] = features
    labels = torch.LongTensor(labels)

    splitted_idx = load_fixed_splits(dataset, data_split_path, split_idx)
    idx_train, idx_val, idx_test = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    return g, labels, idx_train, idx_val, idx_test


def load_bgnn_data(dataset, dataset_path, split_idx):
    data_path = Path.cwd().joinpath(dataset_path, f"{dataset}")

    g, X, y, cat_features, masks = read_input(data_path)
    train_mask, val_mask, test_mask = (
        masks[str(split_idx)]["train"],
        masks[str(split_idx)]["val"],
        masks[str(split_idx)]["test"],
    )

    encoded_X = X.copy()
    if cat_features is not None and len(cat_features):
        encoded_X = encode_cat_features(
            encoded_X, y, cat_features, train_mask, val_mask, test_mask
        )
    encoded_X = normalize_features(encoded_X, train_mask, val_mask, test_mask)
    encoded_X = replace_na(encoded_X, train_mask)
    features, labels = pandas_to_torch(encoded_X, y)

    g = g.remove_self_loop().add_self_loop()
    g.ndata["feat"] = features
    labels = labels.long()

    idx_train = torch.LongTensor(train_mask)
    idx_val = torch.LongTensor(val_mask)
    idx_test = torch.LongTensor(test_mask)
    return g, labels, idx_train, idx_val, idx_test


""" For NonHom"""
dataset_drive_url = {"pokec": "1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y"}
splits_drive_url = {"pokec": "1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_"}


def load_penn94_mat(data_path):
    mat = scipy.io.loadmat(data_path)
    A = mat["A"]
    metadata = mat["local_info"]

    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    metadata = metadata.astype(np.int)

    # make features into one-hot encodings
    feature_vals = np.hstack((np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
        features = np.hstack((features, feat_onehot))

    g = dgl.graph((edge_index[0], edge_index[1]))
    g = dgl.to_bidirected(g)

    features = torch.tensor(features, dtype=torch.float)
    labels = torch.tensor(metadata[:, 1] - 1)  # gender label, -1 means unlabeled
    return g, features, labels


def load_pokec_mat(data_path):
    if not os.path.exists(data_path):
        gdd.download_file_from_google_drive(
            file_id=dataset_drive_url["pokec"], dest_path=data_path, showsize=True
        )

    fulldata = scipy.io.loadmat(data_path)
    edge_index = torch.tensor(fulldata["edge_index"], dtype=torch.long)
    g = dgl.graph((edge_index[0], edge_index[1]))
    g = dgl.to_bidirected(g)

    features = torch.tensor(fulldata["node_feat"]).float()
    labels = fulldata["label"].flatten()
    return g, features, labels


class NCDataset(object):
    def __init__(self, name, root):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/

        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type="random", train_prop=0.5, valid_prop=0.25):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == "random":
            ignore_negative = False if self.name == "ogbn-proteins" else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label,
                train_prop=train_prop,
                valid_prop=valid_prop,
                ignore_negative=ignore_negative,
            )
            split_idx = {"train": train_idx, "valid": valid_idx, "test": test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, len(self))


def load_fixed_splits(dataset, data_split_path="", split_idx=0):
    if not os.path.exists(data_split_path):
        assert dataset in splits_drive_url.keys()
        gdd.download_file_from_google_drive(
            file_id=splits_drive_url[dataset], dest_path=data_split_path, showsize=True
        )

    splits_lst = np.load(data_split_path, allow_pickle=True)
    splits = splits_lst[split_idx]

    for key in splits:
        if not torch.is_tensor(splits[key]):
            splits[key] = torch.as_tensor(splits[key])

    return splits


"""For BGNN """


def pandas_to_torch(*args):
    return [torch.from_numpy(arg.to_numpy(copy=True)).float().squeeze() for arg in args]


def read_input(input_folder):
    X = pd.read_csv(f"{input_folder}/X.csv")
    y = pd.read_csv(f"{input_folder}/y.csv")

    categorical_columns = []
    if os.path.exists(f"{input_folder}/cat_features.txt"):
        with open(f"{input_folder}/cat_features.txt") as f:
            for line in f:
                if line.strip():
                    categorical_columns.append(line.strip())

    cat_features = None
    if categorical_columns:
        columns = X.columns
        cat_features = np.where(columns.isin(categorical_columns))[0]

        for col in list(columns[cat_features]):
            X[col] = X[col].astype(str)

    gs, _ = load_graphs(f"{input_folder}/graph.dgl")
    graph = gs[0]

    with open(f"{input_folder}/masks.json") as f:
        masks = json.load(f)

    return graph, X, y, cat_features, masks


def normalize_features(X, train_mask, val_mask, test_mask):
    min_max_scaler = preprocessing.MinMaxScaler()
    A = X.to_numpy(copy=True)
    A[train_mask] = min_max_scaler.fit_transform(A[train_mask])
    A[val_mask + test_mask] = min_max_scaler.transform(A[val_mask + test_mask])
    return pd.DataFrame(A, columns=X.columns).astype(float)


def replace_na(X, train_mask):
    if X.isna().any().any():
        return X.fillna(X.iloc[train_mask].min() - 1)
    return X


def encode_cat_features(X, y, cat_features, train_mask, val_mask, test_mask):
    enc = CatBoostEncoder()
    A = X.to_numpy(copy=True)
    b = y.to_numpy(copy=True)
    A[np.ix_(train_mask, cat_features)] = enc.fit_transform(
        A[np.ix_(train_mask, cat_features)], b[train_mask]
    )
    A[np.ix_(val_mask + test_mask, cat_features)] = enc.transform(
        A[np.ix_(val_mask + test_mask, cat_features)]
    )
    A = A.astype(float)
    return pd.DataFrame(A, columns=X.columns)


def has_all_reversed_edges(g):
    src, dst = g.edges(order='srcdst')
    src_reverse, dst_reverse = g.reverse().edges(order='srcdst')
    return torch.equal(src, src_reverse) and torch.equal(dst, dst_reverse)


def has_self_loop_every_node(g):
    # Get the list of edges
    src, dst = g.edges()

    # Check if each node has a self-loop
    nodes_with_self_loop = set(src.tolist()) & set(dst.tolist())
    all_nodes = set(range(g.number_of_nodes()))

    return nodes_with_self_loop == all_nodes


def is_not_multigraph(g):
    return not g.is_multigraph
