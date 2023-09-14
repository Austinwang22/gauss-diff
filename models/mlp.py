import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn import LeakyReLU

from .utils import get_act


class MLP(nn.Module):
    def __init__(self, config):
        """Simple MLP

        Args:
            config (namespace): namespace of model config
        """
        super(MLP, self).__init__()
        layers = config.model.layers
        self.act = get_act(config.model.act)
        self.layers = nn.ModuleList([
            nn.Linear(in_ch, out_ch) for in_ch, out_ch in zip(layers, layers[1:])
        ])

    def forward(self, x, t):
        if t.shape == torch.Size([]):
            t = torch.ones((x.shape[0],), device=x.device) * t
        y = torch.cat([x, t[:, None]], dim=-1)
        for i, layer in enumerate(self.layers):
            y = layer(y)
            if i < len(self.layers):
                y = self.act(y)
        return y



class VEPrecond(nn.Module):
    def __init__(self, config, sigma_min=0.02, sigma_max=100):
        super().__init__()
        self.model = MLP(config)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def forward(self, x, sigma, labels=None):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32)
        c_out = sigma
        c_noise = (0.5 * sigma).log()

        F_x = self.model(x, c_noise)
        D_x = x + c_out * F_x
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
    

class Network(nn.Module):

    def __init__(self, device, dimension=2):
        '''
        Time dependent MLP to learn the score of a distribution.
        :param sde: SDE used to perturb the dataset
        :param device: device used
        :param dimension: dimension of each element in dataset (e.g. R^2)
        '''
        super().__init__()

        self.lin1 = Linear(dimension + 1, 256)
        self.lrelu1 = LeakyReLU()

        self.lin2 = Linear(256, 256)
        self.lrelu2 = LeakyReLU()

        self.lin3 = Linear(256, 256)
        self.lrelu3 = LeakyReLU()

        self.lin4 = Linear(256, 256)
        self.lrelu4 = LeakyReLU()

        self.lin5 = Linear(256, dimension)

        self.device = device

    def forward(self, x, t):

        x = x.to(self.device)

        x = torch.cat([x, t[:, None]], dim=-1)

        x = self.lin1(x)
        x = self.lrelu1(x)

        x = self.lin2(x)
        x = self.lrelu2(x)

        x = self.lin3(x)
        x = self.lrelu3(x)

        x = self.lin4(x)
        x = self.lrelu4(x)

        x = self.lin5(x)

        return x
