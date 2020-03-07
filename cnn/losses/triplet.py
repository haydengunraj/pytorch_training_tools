import torch
import torch.nn.functional as F


def create(config):
    return TripletLoss(**config)


class TripletLoss:
    """Triplet loss, which assumes the input is a repeating
    sequence of anchor, positive, and negative embeddings"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, triplets):
        anchor = triplets[::3]
        positive = triplets[1::3]
        negative = triplets[2::3]
        dist = torch.norm(anchor - positive, p=2, dim=1) - torch.norm(anchor - negative, p=2, dim=1) + self.alpha
        loss = torch.mean(F.relu(dist))
        return loss
