import torch.nn as nn
import torch.nn.functional as F


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, model, logits, labels):
        embeddings = F.normalize(logits) * model.prompts
        return self.cross_entropy_loss(embeddings @ model.prototypes.T, labels)


class BaselineLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, model, logits, labels):
        return self.cross_entropy_loss(logits, labels)
