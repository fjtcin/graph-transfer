from torch_geometric.datasets import Planetoid
import scipy.sparse as ss
import numpy as np
from dataloader import SparseGraph, save_sparse_graph_to_npz

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.cluster import FeatureAgglomeration

from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.neural_network import BernoulliRBM

N_DIM = 128

def reduction(feature, method, scaler, train_mask, test_mask, label=None):
    x_train, x_test = feature[train_mask], feature[test_mask]
    y_train = label[train_mask] if label is not None else None
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    method.fit(x_train, y_train)

    embedding = np.zeros((feature.shape[0], N_DIM))
    embedding[train_mask] = method.transform(x_train)
    embedding[test_mask] = method.transform(x_test)
    return embedding

def main():
    dataset = Planetoid('data/planetoid', 'cora')
    data = dataset[0]

    identity = FunctionTransformer()
    stdscalar = StandardScaler()

    pca = PCA(n_components=N_DIM)
    kpca = KernelPCA(n_components=N_DIM)
    tsvd = TruncatedSVD(n_components=N_DIM)
    isomap = Isomap(n_components=N_DIM)
    lle = LocallyLinearEmbedding(n_components=N_DIM)
    grp = GaussianRandomProjection(n_components=N_DIM)
    srp = SparseRandomProjection(n_components=N_DIM)
    agg = FeatureAgglomeration(n_clusters=N_DIM)

    pls = PLSRegression(n_components=N_DIM)
    nca = NeighborhoodComponentsAnalysis(n_components=N_DIM)
    rbm = BernoulliRBM(n_components=N_DIM)

    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(data.y.reshape(-1, 1))

    reduced_features = reduction(data.x, nca, identity, data.train_mask, data.val_mask | data.test_mask, data.y)

    print("Original data shape:", data.x.shape)
    print("Reduced data shape:", reduced_features.shape)

    edge_list = data.edge_index.numpy()
    ones = np.ones(data.num_edges)
    adj_matrix = ss.csr_matrix((ones, (edge_list[0], edge_list[1])))

    sparse_graph = SparseGraph(adj_matrix, attr_matrix=reduced_features, labels=data.y.numpy())
    save_sparse_graph_to_npz(f'data/cora.npz', sparse_graph)

if __name__ == "__main__":
    main()
