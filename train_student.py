import argparse
import copy
import numpy as np
import torch
from torch import optim
from torch import nn
from pathlib import Path
from models import Model
from criterion import CosineSimilarityLoss, CustomKLDivLoss
from dataloader import load_data
from utils import (
    get_logger,
    get_evaluator,
    set_seed,
    get_training_config,
    check_writable,
    check_readable,
    compute_min_cut_loss,
    graph_split,
    feature_prop,
)
from train_and_eval import distill_run_transductive, distill_run_inductive


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
    parser.add_argument("--teacher", type=str, default="SAGE", help="Teacher model")
    parser.add_argument("--student", type=str, default="MLP", help="Student model")
    parser.add_argument(
        "--prompts_dim", type=int, default=256, help="Model prompts dimensions"
    )

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
    parser.add_argument(
        "--upstream_feature_aug_k",
        type=int,
        default=0,
    )

    """Distiall"""
    parser.add_argument(
        "--lamb",
        type=float,
        default=0,
        help="Parameter balances loss from hard labels and teacher outputs, take values in [0, 1]",
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

    args.out_t_path = args.output_path  # Path to load teacher outputs

    if args.feature_aug_k > 0:
        args.output_path = args.output_path + f"/aug_hop_{args.feature_aug_k}"

    if args.upstream_feature_aug_k > 0:
        # Teacher may or may not have augmented features.
        args.out_t_path = args.out_t_path + f"/aug_hop_{args.upstream_feature_aug_k}"

    if args.exp_setting == "tran":
        output_dir = Path.cwd().joinpath(
            args.output_path,
            "transductive",
            args.dataset,
            f"{args.teacher}_{args.student}",
            f"seed_{args.seed}",
        )
        out_t_dir = Path.cwd().joinpath(
            args.out_t_path,
            "transductive",
            args.dataset,
            args.teacher,
            f"seed_{args.seed}",
        )
    else:
        output_dir = Path.cwd().joinpath(
            args.output_path,
            "inductive",
            f"split_rate_{args.split_rate}",
            args.dataset,
            f"{args.teacher}_{args.student}",
            f"seed_{args.seed}",
        )
        out_t_dir = Path.cwd().joinpath(
            args.out_t_path,
            "inductive",
            f"split_rate_{args.split_rate}",
            args.dataset,
            args.teacher,
            f"seed_{args.seed}",
        )
    args.output_dir = output_dir

    check_writable(output_dir, overwrite=False)
    check_readable(out_t_dir)

    logger = get_logger(output_dir.joinpath("log"), args.console_log, args.log_level)
    logger.info(f"out_t_dir: {out_t_dir}")

    """ Load data and model config"""
    g, labels, idx_train, idx_val, idx_test = load_data(
        args.dataset,
        args.data_path,
        split_idx=args.split_idx,
        seed=args.seed,
    )
    logger.info(f"Total {g.number_of_nodes()} nodes, {g.number_of_edges()} edges.")

    feats = g.ndata["feat"]
    args.feat_dim = g.ndata["feat"].shape[1]
    label_dim = labels.max().item() + 1
    labels = nn.functional.one_hot(labels).float()

    if 0 < args.feature_noise <= 1:
        feats = (
            1 - args.feature_noise
        ) * feats + args.feature_noise * torch.randn_like(feats)

    """ Model config """
    conf = {}
    if args.model_config_path is not None:
        conf = get_training_config(
            args.model_config_path, f"{f'GA{args.feature_aug_k}' if args.feature_aug_k else ''}{args.student}", args.dataset
        )  # Note: student config
    conf = dict(args.__dict__, **conf)
    conf["device"] = device
    logger.info(f"conf: {conf}")

    """ Model init """
    model = Model(conf)
    # model.prompts = torch.nn.Parameter(torch.randn(label_dim, conf["prompts_dim"]).to(device))
    model.prompts = torch.nn.Parameter(torch.from_numpy(np.load(out_t_dir.joinpath("prompts.npz"))["arr_0"]).to(device), requires_grad=False)
    model.p = torch.nn.Parameter(torch.ones(1, conf["feat_dim"]).to(device), requires_grad=False)
    logger.info(f"prompts.requires_grad = {model.prompts.requires_grad}, p.requires_grad = {model.p.requires_grad}")
    optimizer = optim.Adam(
        model.parameters(), lr=conf["learning_rate"], weight_decay=conf["weight_decay"]
    )
    criterion_l = CosineSimilarityLoss()
    criterion_t = CustomKLDivLoss()
    evaluator = get_evaluator(conf["dataset"])

    """Load teacher model output"""
    out_t = torch.from_numpy(np.load(out_t_dir.joinpath("out.npz"))["arr_0"])

    """Data split and run"""
    loss_and_score = []
    if args.exp_setting == "tran":
        idx_l = idx_train
        idx_t = torch.cat([idx_train, idx_val, idx_test])
        distill_indices = (idx_l, idx_t, idx_val, idx_test)

        # propagate node feature
        if args.feature_aug_k > 0:
            feats = feature_prop(feats, g, args.feature_aug_k)

        out, score_val, score_test = distill_run_transductive(
            conf,
            model,
            feats,
            labels,
            out_t,
            distill_indices,
            criterion_l,
            criterion_t,
            evaluator,
            optimizer,
            logger,
            loss_and_score,
        )
        score_lst = [score_test]

    else:
        # Create inductive split
        obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = graph_split(
            idx_train, idx_val, idx_test, args.split_rate, args.seed
        )
        obs_idx_l = obs_idx_train
        obs_idx_t = torch.cat([obs_idx_train, obs_idx_val, obs_idx_test])
        distill_indices = (
            obs_idx_l,
            obs_idx_t,
            obs_idx_val,
            obs_idx_test,
            idx_obs,
            idx_test_ind,
        )

        # propagate node feature. The propagation for the observed graph only happens within the subgraph obs_g
        if args.feature_aug_k > 0:
            obs_g = g.subgraph(idx_obs)
            obs_feats = feature_prop(feats[idx_obs], obs_g, args.feature_aug_k)
            feats = feature_prop(feats, g, args.feature_aug_k)
            feats[idx_obs] = obs_feats

        out, score_val, score_test_tran, score_test_ind = distill_run_inductive(
            conf,
            model,
            feats,
            labels,
            out_t,
            distill_indices,
            criterion_l,
            criterion_t,
            evaluator,
            optimizer,
            logger,
            loss_and_score,
        )
        score_lst = [score_test_tran, score_test_ind]

    logger.info(f"# params {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    """ Saving student outputs """
    out_np = out.numpy(force=True)
    np.savez(output_dir.joinpath("out"), out_np)

    """ Saving loss curve and model """
    if args.save_results:
        # Loss curves
        loss_and_score = np.array(loss_and_score)
        np.savez(output_dir.joinpath("loss_and_score"), loss_and_score)

        # Model
        torch.save(model.state_dict(), output_dir.joinpath("model.pth"))

    """ Saving min-cut loss"""
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
