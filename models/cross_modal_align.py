import torch
import torch.nn as nn

class CrossModalAlignLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.loss_fn = nn.TripletMarginLoss(margin=margin)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)