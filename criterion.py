import torch.nn as nn
import torch.nn.functional as F


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, logits, prompts, labels):
        logits_n = F.normalize(logits)
        return self.cross_entropy_loss(logits_n @ prompts.mT, labels)


class BaselineLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, logits, prompts, labels):
        return self.cross_entropy_loss(logits, labels)
