from torch import nn
import torch
from torch.nn import functional as F


def cosine_sim(im, s):
    """
    Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


class ContrastiveLoss(nn.Module):
    """
    ontrastive Loss between 2 groups of embeddings

    inputs shape: (batch, embed_dim)
    """

    def __init__(self, use_cuda, margin=0.2, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.sim = cosine_sim
        self.use_cuda = use_cuda
        self.max_violation = max_violation

    def forward(self, im, s, device):
        im_norm = F.normalize(im)
        s_norm = F.normalize(s)
        scores = self.sim(im_norm, s_norm)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)
        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_im = (self.margin + scores - d2).clamp(min=0)
        mask = torch.eye(scores.size(0)) > .5
        if self.use_cuda:
            mask = mask.to(device)
        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)

        # with hard negative
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()
