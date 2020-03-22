import torch
import torch.nn as nn

from .embednet import EmbedNet


def create(config):
    return CompareNet(**config)


class CompareNet(nn.Module):
    def __init__(self, obj_emb_size, img_emb_size, clf_hidden_size,
                 obj_pool_size=(1, 1), img_pool_size=(1, 1), pretrained=True):
        super().__init__()
        self.object_embedder = EmbedNet(obj_emb_size, pool_size=obj_pool_size, pretrained=pretrained)
        self.image_embedder = EmbedNet(img_emb_size, pool_size=img_pool_size, pretrained=pretrained)
        self.classifier = nn.Sequential(
            nn.Linear(img_emb_size + obj_emb_size, clf_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(clf_hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, image, obj_image):
        img_emb = self.image_embedder(image)
        obj_emb = self.object_embedder(obj_image)
        cat = torch.cat((img_emb, obj_emb), dim=1)
        score = self.classifier(cat)
        return score
