import os
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.distributions.multivariate_normal import MultivariateNormal

from models.mlp import MLP, Network
from models.utils import get_score_fn
from sampling import get_sampling_fn

from utils.sde_lib import VPSDE, VESDE, EDMSDE
from utils.helper import dict2namespace, save_ckpt
from utils.datasets import Gaussian, GaussianMixture
from matplotlib import pyplot as plt
from plots import plot_data
from utils.posterior import compute_posterior


def subprocess(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    config = dict2namespace(config)
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    print('Using {} device'.format(device))

    basedir = config.log.basedir
    basedir = os.path.join('exp', basedir)
    os.makedirs(basedir, exist_ok=True)
    figsdir = os.path.join(basedir, 'figs')
    os.makedirs(figsdir, exist_ok=True)
    ckptsdir = os.path.join(basedir, 'ckpts')
    os.makedirs(ckptsdir, exist_ok=True)

    model = Network(device).to(device)
    print("Loading preset weights from {}".format(config.model.ckpt_path))
    model.load_state_dict(torch.load(config.model.ckpt_path))

    mean = torch.tensor(config.data.mean, dtype=torch.float32)
    std = torch.tensor(config.data.std, dtype=torch.float32)
    type = config.data.type
    
    if type == 'gaussian':
        prior = Gaussian(num_samples=config.data.num_samples,
                        mean=mean, std=std)
    elif type == 'mixture':
        prior = GaussianMixture(num_samples=config.data.num_samples,
                                mean=mean, std=std)

    x = torch.load(config.data.x_path).to(device)
    H = torch.tensor(config.data.H, device=device)
    std_y = config.data.std_y
    y = config.data.y

    if config.model.sde == 'VP':
        sde = VPSDE()
    elif config.model.sde == 'VE':
        sde = VESDE()

    num_samples = config.sampling.num_samples
    sample_shape = (num_samples, config.data.dim)

    def scaler(x):
        return x

    sample_fn = get_sampling_fn(
        config, sde, shape=sample_shape, inverse_scaler=scaler, eps=config.sampling.eps, device=device)
    
    start_x = torch.randn(sample_shape, device=device)

    # approx = pigdm_approximation(sde, model, start_x[0], 1., num_samples, device, continuous=config.train.continuous)
    # true = true_x0_xt_density(sde, prior, start_x[0], 1., num_samples)

    out_dir = os.path.join(basedir, 'compare')

    samples = sample_fn(model, y, H, std_y, prior=prior, start_x=start_x, out_dir=out_dir)

    posterior = compute_posterior(prior, y, H, std_y, config.sampling.posterior_samples)

    samples_path = os.path.join(figsdir, 'pigdm_samples.png')
    posterior_path = os.path.join(figsdir, 'posterior.png')
    prior_path = os.path.join(figsdir, 'prior.png')
    # approx_path = os.path.join(figsdir, 'approx.png')
    # true_path = os.path.join(figsdir, 'true.png')

    plot_data([samples.detach().cpu()], 'Samples', samples_path)
    plot_data([posterior.data.detach().cpu()], 'Posterior', posterior_path)
    plot_data([prior.data.detach().cpu()], 'Prior', prior_path)
    # plot_data([approx.data.detach().cpu()], 'Approximated p(x_0 | x_t)', approx_path)
    # plot_data([true.data.detach().cpu()], 'True p(x_0 | x_t)', true_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/gaussian.yml')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ckpt', type=str, default='none')
    args = parser.parse_args()
    subprocess(args)