import argparse
import copy
import numpy as np
import scipy.sparse as ss
from utils import set_seed
from dataloader import SparseGraph, save_sparse_graph_to_npz

import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


class SemiSupervisedAutoencoder(nn.Module):
    def __init__(self, num_feats, num_classes, dropout_autoencoder, dropout_MLP):
        super().__init__()
        assert num_feats > 128

        # Shared encoder
        num_feats_log = (num_feats - 1).bit_length() - 1
        encoder_lst = [nn.Linear(num_feats, 1<<num_feats_log)]
        for i in range(num_feats_log, 7, -1):
            encoder_lst += [nn.ReLU(), nn.Dropout(dropout_autoencoder), nn.Linear(1<<i, 1<<i-1)]
        self.encoder = nn.Sequential(*encoder_lst)

        # Decoder for reconstruction task
        decoder_lst = []
        for i in range(7, num_feats_log):
            decoder_lst += [nn.Linear(1<<i, 1<<i+1), nn.ReLU()]
        decoder_lst += [nn.Linear(1<<num_feats_log, num_feats)]
        self.decoder = nn.Sequential(*decoder_lst)

        # Classifier for classification task
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout_MLP),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        reconstructed = self.decoder(x)
        class_probs = self.classifier(x)
        return reconstructed, class_probs


class CombinedLoss(nn.Module):
    def __init__(self, lamb=.5):
        super().__init__()
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.lamb = lamb

    def forward(self, reconstructed, original, class_probs, labels):
        reconstruction_loss = self.kl_div_loss(reconstructed.log_softmax(dim=1), original.log_softmax(dim=1))
        classification_loss = self.cross_entropy_loss(class_probs, labels)
        return self.lamb * reconstruction_loss + (1 - self.lamb) * classification_loss


def AccuracyEvaluator():
    def evaluator(class_probs, labels):
        return class_probs.argmax(dim=-1).eq(labels).float().mean().item()
    return evaluator


def train(model, feats, labels, criterion, optimizer):
    model.train()
    reconstructed, class_probs = model(feats)
    loss = criterion(reconstructed, feats, class_probs, labels)
    loss_val = loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss_val


def eval(model, feats, labels, criterion, evaluator):
    model.eval()
    with torch.no_grad():
        reconstructed, class_probs = model(feats)
        loss = criterion(reconstructed, feats, class_probs, labels)
        score = evaluator(class_probs, labels)
        return loss.item(), score


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch DGL implementation")
    parser.add_argument("--device", type=int, default=0, help="CUDA device, -1 means CPU")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'pubmed'])
    parser.add_argument("--data_path", type=str, default="./data", help="Path to data")
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--dropout_autoencoder", type=float, default=0)
    parser.add_argument("--dropout_MLP", type=float, default=0)
    parser.add_argument("--max_epoch", type=int, default=1000, help="Evaluate once per how many epochs")
    parser.add_argument("--eval_interval", type=int, default=1, help="Evaluate once per how many epochs")
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Early stop is the score on validation set does not improve for how many epochs",
    )
    parser.add_argument(
        "--lamb",
        type=float,
        default=0,
        help="Parameter balances loss from autoencoder and MLP",
    )
    return parser.parse_args()


def run(args):
    set_seed(args.seed)
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device("cuda:" + str(args.device))
    else:
        device = "cpu"
    dataset = Planetoid(args.data_path, args.dataset, transform=T.ToDevice(device))
    data = dataset[0]

    # model = SemiSupervisedAutoencoder(data.x.size(1), data.y.max().item()+1, args.dropout_autoencoder, args.dropout_MLP).to(device)
    # criterion = CombinedLoss(args.lamb)
    # optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # evaluator = AccuracyEvaluator()

    # best_score_val, count = 0, 0
    # for epoch in range(1, args.max_epoch + 1):
    #     train(model, data.x[data.train_mask], data.y[data.train_mask], criterion, optimizer)
    #     if epoch % args.eval_interval == 0:
    #         loss_val, score_val = eval(model, data.x[data.val_mask], data.y[data.val_mask], criterion, evaluator)
    #         if score_val >= best_score_val:
    #             best_score_val = score_val
    #             state = copy.deepcopy(model.state_dict())
    #             count = 0
    #         else:
    #             count += 1
    #         if count == args.patience:
    #             break

    # model.load_state_dict(state)
    # loss_test, score_test = eval(model, data.x[data.test_mask], data.y[data.test_mask], criterion, evaluator)
    # print(f"loss: {loss_test:.4f}, score: {score_test:.4f}")
    # model.eval()
    # with torch.no_grad():
    #     z = model.encoder(data.x)

    edge_list = data.edge_index.numpy(force=True)
    ones = np.ones(data.num_edges)
    adj_matrix = ss.csr_matrix((ones, (edge_list[0], edge_list[1])))

    sparse_graph = SparseGraph(adj_matrix, attr_matrix=data.x.numpy(force=True), labels=data.y.numpy(force=True))
    save_sparse_graph_to_npz(f'data/{args.dataset}.npz', sparse_graph)


def main():
    args = get_args()
    run(args)


if __name__ == "__main__":
    main()
