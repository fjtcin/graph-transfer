import copy
import torch
import dgl


def train(model, dataloader, criterion, evaluator, optimizer):
    model.train()
    total_loss, total_logits, total_labels = 0, [], []
    for input_nodes, output_nodes, blocks in dataloader:
        input_features = blocks[0].srcdata['feat']
        output_labels = blocks[-1].dstdata['label']

        logits = model(blocks, input_features * model.p)
        loss = criterion(model, logits, output_labels)

        total_loss += loss.item()
        total_logits.append(logits)
        total_labels.append(output_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_logits = torch.cat(total_logits, dim=0)
    total_labels = torch.cat(total_labels, dim=0)
    score = evaluator(model, total_logits, total_labels)
    return total_loss / len(dataloader), score


@torch.no_grad()
def evaluate(model, dataloader, criterion, evaluator):
    model.eval()  # Set model to evaluation mode
    total_loss, total_logits, total_labels = 0, [], []
    for input_nodes, output_nodes, blocks in dataloader:
        input_features = blocks[0].srcdata['feat']
        output_labels = blocks[-1].dstdata['label']

        logits = model(blocks, input_features * model.p, inference=True)
        loss = criterion(model, logits, output_labels)

        total_loss += loss.item()
        total_logits.append(logits)
        total_labels.append(output_labels)

    total_logits = torch.cat(total_logits, dim=0)
    total_labels = torch.cat(total_labels, dim=0)
    score = evaluator(model, total_logits, total_labels)
    return total_loss / len(dataloader), score


def run_transductive(
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
):
    """
    Train and eval under the transductive setting.
    The train/valid/test split is specified by `indices`.
    The input graph is assumed to be large. Thus, SAGE is used for GNNs, mini-batch is used for MLPs.

    loss_and_score: Stores losses and scores.
    """
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(model.num_layers)
    dataloader_train = dgl.dataloading.DataLoader(
        g,
        indices[0],
        sampler,
        device=conf["device"],
        batch_size=conf["batch_size"],
        use_uva=True
    )
    dataloader_val = dgl.dataloading.DataLoader(
        g,
        indices[1],
        sampler,
        device=conf["device"],
        batch_size=conf["batch_size"],
        use_uva=True
    )
    dataloader_test = dgl.dataloading.DataLoader(
        g,
        indices[2],
        sampler,
        device=conf["device"],
        batch_size=conf["batch_size"],
        use_uva=True
    )

    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        loss, score = train(model, dataloader_train, criterion, evaluator, optimizer)
        logger.debug(f"Ep {epoch:3d} | loss_train: {loss:.4f} | acc_train: {score:.4f}")

        if epoch % conf["eval_interval"] == 0:
            loss, score = evaluate(model, dataloader_val, criterion, evaluator)
            logger.debug(f"Ep {epoch:3d} | loss_val: {loss:.4f} | acc_val: {score:.4f}")

            if score >= best_score_val:
                best_epoch = epoch
                best_score_val = score
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    loss, score = evaluate(model, dataloader_test, criterion, evaluator)
    logger.info(f"Best valid model at epoch: {best_epoch: 3d}, score_val: {best_score_val :.4f}, score_test: {score :.4f}")
    return None, loss, score

def run_inductive(*args, **kwargs):
    pass
