import torch
import torch.nn as nn


class OUSDE(nn.Module):

    def __init__(self, drift, shift, diff):
        super().__init__()
        # Scalar parameter.
        self.theta = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        self.drift = drift
        self.shift = shift
        self.diff = diff

    def f(self, t, y):
        return self.drift * (y - self.shift)

    def g(self, t, y):
        return self.diff * torch.ones_like(y)


class ReverseSDE(nn.Module):
    def __init__(self, drift, shift, diff,
                 ratio,
                 score):
        super().__init__()
        self.drift = drift
        self.shift = shift
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        self.diff = diff
        self.ratio = ratio
        self.score = score

    def f(self, t, y):
        res = self.drift * (y - self.shift) - self.diff * \
            self.diff * self.score(1 - t, y)
        return - res

    def g(self, t, y):
        return self.diff * torch.ones_like(y)
