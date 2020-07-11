import torch
import torch.nn as nn
import torch.nn.functional as F


class Transormers(nn.Module):

    def __init__(self):
        super(Transormers, self).__init__()
        self.src_mask = None
        self.pos_encoder = ""


"""
    Attention models require  positional encoding 
    going with default implmentation from paper using sine and cos funtions

    https://arxiv.org/pdf/1706.03762.pdf
"""


class PosEn(nn.Module):

    def __init__(self):
        pass
