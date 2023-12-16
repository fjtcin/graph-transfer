import argparse
import copy
import numpy as np
import torch
from torch import optim
from torch import nn
from pathlib import Path
from models import Model
from criterion import BaselineLoss
from dataloader import load_data
from utils import (
    get_logger,
    get_evaluator,
    set_seed,
    get_training_config,
    check_writable,
    compute_min_cut_loss,
    graph_split,
    feature_prop,
)
from train_and_eval import run_transductive, run_inductive


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch DGL implementation")
    parser.add_argument("--device", type=int, default=0, help="CUDA device, -1 means CPU")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--log_level",
        type=int,
        default=20,
        help="Logger levels for run {10: DEBUG, 20: INFO, 30: WARNING}",
    )
    parser.add_argument(
        "--console_log",
        action="store_true",
        help="Set to True to display log info in console",
    )
    parser.add_argument(
        "--output_path", type=str, default="outputs", help="Path to save outputs"
    )
    parser.add_argument(
        "--num_exp", type=int, default=1, help="Repeat how many experiments"
    )
    parser.add_argument(
        "--exp_setting",
        type=str,
        default="tran",
        choices=["tran", "ind"],
        help="transductive or inductive",
    )
    parser.add_argument(
        "--eval_interval", type=int, default=1, help="Evaluate once per how many epochs"
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Set to True to save the loss curves, trained model, and min-cut loss for the transductive setting",
    )

    """Dataset"""
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to data")
    parser.add_argument(
        "--split_idx",
        type=int,
        default=0,
        help="For Non-Homo datasets only, one of [0,1,2,3,4]",
    )

    """Model"""
    parser.add_argument(
        "--model_config_path",
        type=str,
        default="./train.conf.yaml",
        help="Path to model configeration",
    )
    parser.add_argument("--model", type=str, default="SAGE")

    """Optimization"""
    parser.add_argument(
        "--max_epoch", type=int, default=1000, help="Evaluate once per how many epochs"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Early stop is the score on validation set does not improve for how many epochs",
    )

    """Ablation"""
    parser.add_argument(
        "--feature_noise",
        type=float,
        default=0,
        help="add white noise to features for analysis, value in [0, 1] for noise level",
    )
    parser.add_argument(
        "--split_rate",
        type=float,
        default=0.2,
        help="Rate for graph split, see comment of graph_split for more details",
    )
    parser.add_argument(
        "--compute_min_cut",
        action="store_true",
        help="Set to True to compute and store the min-cut loss",
    )
    parser.add_argument(
        "--feature_aug_k",
        type=int,
        default=0,
        help="Augment node futures by aggregating feature_aug_k-hop neighbor features",
    )

    args = parser.parse_args()
    return args


def run(args):
    """
    Returns:
    score_lst: a list of evaluation results on test set.
    len(score_lst) = 1 for the transductive setting.
    len(score_lst) = 2 for the inductive/production setting.
    """

    """ Set seed, device, and logger """
    set_seed(args.seed)
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device("cuda:" + str(args.device))
    else:
        device = "cpu"

    if args.feature_noise != 0:
        args.output_path = args.output_path + f"/noise_{args.feature_noise}"

    if args.feature_aug_k > 0:
        args.output_path = args.output_path + f"/aug_hop_{args.feature_aug_k}"

    if args.exp_setting == "tran":
        output_dir = Path.cwd().joinpath(
            args.output_path,
            "transductive",
            args.dataset,
            args.model,
            f"seed_{args.seed}",
        )
    else:
        output_dir = Path.cwd().joinpath(
            args.output_path,
            "inductive",
            f"split_rate_{args.split_rate}",
            args.dataset,
            args.model,
            f"seed_{args.seed}",
        )
    args.output_dir = output_dir

    check_writable(output_dir, overwrite=False)
    logger = get_logger(output_dir.joinpath("log"), args.console_log, args.log_level)

    """ Load data """
    g, labels, idx_train, idx_val, idx_test = load_data(
        args.dataset,
        args.data_path,
        split_idx=args.split_idx,
        seed=args.seed,
    )
    logger.info(f"Total {g.number_of_nodes()} nodes, {g.number_of_edges()} edges.")

    feats = g.ndata["feat"]
    args.feat_dim = feats.shape[1]
    label_dim = labels.max().item() + 1
    args.prompts_dim = label_dim

    if 0 < args.feature_noise <= 1:
        feats = (
            1 - args.feature_noise
        ) * feats + args.feature_noise * torch.randn_like(feats)

    """ Model config """
    conf = {}
    if args.model_config_path is not None:
        conf = get_training_config(args.model_config_path, f"{f'GA{args.feature_aug_k}' if args.feature_aug_k else ''}{args.model}", args.dataset)
    conf = dict(args.__dict__, **conf)
    conf["device"] = device
    logger.info(f"conf: {conf}")

    """ Model init """
    model = Model(conf)
    model.prompts = torch.empty(0)
    model.p = torch.ones(1, conf["feat_dim"]).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=conf["learning_rate"], weight_decay=conf["weight_decay"]
    )
    criterion = BaselineLoss()
    evaluator = get_evaluator(conf["dataset"], True)

    """ Data split and run """
    loss_and_score = []
    if args.exp_setting == "tran":
        indices = (idx_train, idx_val, idx_test)

        # propagate node feature
        if args.feature_aug_k > 0:
            feats = feature_prop(feats, g, args.feature_aug_k)

        out, score_val, score_test = run_transductive(
            conf,
            model,
            g,
            feats,
            labels,
            indices,
            criterion,
            evaluator,
            optimizer,
            logger,
            loss_and_score,
        )
        score_lst = [score_test]

    else:
        indices = graph_split(idx_train, idx_val, idx_test, args.split_rate, args.seed)

        # propagate node feature. The propagation for the observed graph only happens within the subgraph obs_g
        if args.feature_aug_k > 0:
            idx_obs = indices[3]
            obs_g = g.subgraph(idx_obs)
            obs_feats = feature_prop(feats[idx_obs], obs_g, args.feature_aug_k)
            feats = feature_prop(feats, g, args.feature_aug_k)
            feats[idx_obs] = obs_feats

        out, score_val, score_test_tran, score_test_ind = run_inductive(
            conf,
            model,
            g,
            feats,
            labels,
            indices,
            criterion,
            evaluator,
            optimizer,
            logger,
            loss_and_score,
        )
        score_lst = [score_test_tran, score_test_ind]

    logger.info(f"# params {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    """ Saving loss curve and model """
    if args.save_results:
        # Loss curves
        loss_and_score = np.array(loss_and_score)
        np.savez(output_dir.joinpath("loss_and_score"), loss_and_score)

        # Model
        torch.save(model.state_dict(), output_dir.joinpath("model.pth"))

    """ Saving min-cut loss """
    if args.exp_setting == "tran" and args.compute_min_cut:
        min_cut = compute_min_cut_loss(g, out)
        with open(output_dir.parent.joinpath("min_cut_loss"), "a+") as f:
            f.write(f"{min_cut :.4f}\n")

    return score_lst


def repeat_run(args):
    scores = []
    for seed in range(args.num_exp):
        tmp = copy.deepcopy(args)
        tmp.seed = seed
        scores.append(run(tmp))
    args.output_dir = tmp.output_dir
    scores_np = np.array(scores)
    return scores_np.mean(axis=0), scores_np.std(axis=0)


def main():
    args = get_args()
    if args.num_exp == 1:
        score = run(args)
        score_str = "".join([f"{s : .4f}\t" for s in score])

    elif args.num_exp > 1:
        score_mean, score_std = repeat_run(args)
        score_str = "".join(
            [f"{s : .4f}\t" for s in score_mean] + [f"{s : .4f}\t" for s in score_std]
        )

    with open(args.output_dir.parent.joinpath("exp_results"), "a+") as f:
        f.write(f"{score_str}\n")

    # for collecting aggregated results
    print(score_str)


if __name__ == "__main__":
    main()
