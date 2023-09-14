'''
True score of toy problems
'''
import numpy as np
import torch


class Gscore(object):
    def __init__(self, data_mean, data_var,
                 beta_min=0.1, beta_max=20, device='cpu') -> None:
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.mean = data_mean[None, :].to(device)
        self.var = data_var.to(device)

    def __call__(self, x, t):
        """_summary_

        Args:
            x (tensor): (Batchsize, dim)
            t (tensor): (Batchsize, 1)

        Returns:
            score: (Batchsize, dim)
        """
        log_mean_coeff = -0.25 * t ** 2 * \
            (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        ht = torch.exp(log_mean_coeff)[:, None]
        # print(ht.shape)
        # print(x.shape)
        # print(self.mean.shape)
        vec = ht * self.mean - x    # (B, dim)
        var = ht * ht * self.var + (1 - ht * ht)
        std = torch.sqrt(1 - ht * ht)
        # B, dim
        return - vec / var * std
