import torch
import torch.nn.functional as F
import numpy as np


def triplet_loss(triplets, alpha=0.2):
    """Triplet loss, which assumes the input is a repeating
    sequence of anchor, positive, and negative embeddings"""
    anchor = triplets[::3]
    positive = triplets[1::3]
    negative = triplets[2::3]
    dist = torch.norm(anchor - positive, p=2, dim=1) - torch.norm(anchor - negative, p=2, dim=1) + alpha
    loss = torch.mean(F.relu(dist))
    return loss


def select_triplets(embeddings, indices, num_classes, images_per_class, alpha=0.2):
    """Randomly selects triplets which violate the triplet loss margin
    Based on FaceNet implementation: https://github.com/davidsandberg/facenet"""
    start_idx = 0
    triplets = []
    for i in range(num_classes):
        for j in range(1, images_per_class):
            anchor_idx = start_idx + j - 1
            neg_diffs = embeddings[anchor_idx] - embeddings
            neg_dists_sqr = torch.sum(torch.mul(neg_diffs, neg_diffs), 1)
            for k in range(j, images_per_class):
                positive_idx = start_idx + k
                pos_diff = embeddings[anchor_idx] - embeddings[positive_idx]
                pos_dist_sqr = torch.sum(torch.mul(pos_diff, pos_diff))
                neg_dists_sqr[start_idx:start_idx + images_per_class] = pos_dist_sqr + 1.
                all_neg = torch.where(neg_dists_sqr - pos_dist_sqr < alpha)[0]
                nrof_negs = all_neg.size(0)
                if nrof_negs > 0:
                    rnd_idx = np.random.randint(nrof_negs)
                    negative_idx = int(all_neg[rnd_idx].numpy())
                    triplets.append((indices[anchor_idx], indices[positive_idx], indices[negative_idx]))
        start_idx += images_per_class

    np.random.shuffle(triplets)
    return triplets


class TripletLoss:
    """Wrapper class for triplet loss"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, triplets):
        return triplet_loss(triplets, self.alpha)


class InceptionLoss:
    def __init__(self, loss_type, logit_weight=1.0, aux_weight=1.0, weight=None):
        self.loss_func = getattr(F, loss_type)
        if self.loss_func is None:
            raise ValueError('torch.nn.functional does not define', loss_type)
        self.logit_weight = logit_weight
        self.aux_weight = aux_weight

        self.include_weight = loss_type in ('cross_entropy',)
        self.weight = weight

    def __call__(self, logits, target):
        if isinstance(logits, torch.Tensor):
            return self.loss_func(logits, target)
        elif isinstance(logits, tuple) and len(logits) == 2:
            logits, aux_logits = logits
            return self.logit_weight*self._loss_func(logits, target) \
                + self.aux_weight*self._loss_func(aux_logits, target)
        else:
            raise ValueError('InceptionLoss logits must be a Tensor or a 2-tuple of Tensors')

    def _loss_func(self, logits, target):
        if self.include_weight:
            return self.loss_func(logits, target, weight=self.weight)
        return self.loss_func(logits, target)
