import os
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from models.mlp import MLP, Network
from sampling import get_sampling_fn

from utils.sde_lib import VPSDE, VESDE, EDMSDE
from utils.helper import dict2namespace, save_ckpt
from utils.datasets import Gaussian, GaussianMixture
from matplotlib import pyplot as plt
from plots import plot_data
from utils.posterior import compute_posterior


def observation(H, x, std_y=0.1):
    Hx = torch.matmul(H, x.T)
    mean = torch.zeros_like(Hx)
    std = torch.ones_like(Hx) * std_y
    z = torch.normal(mean, std)
    return Hx + z


def acceptance_probability(curr_sample, candidate, y, H, std_y):
    m1 = (y - torch.matmul(H, curr_sample.T)) ** 2
    m2 = (y - torch.matmul(H, candidate.T)) ** 2
    exp = torch.exp((m1 - m2) / (2. * std_y * std_y))
    accept = torch.minimum(exp, torch.ones_like(exp))
    return accept


def generate_candidate(curr_sample, beta):
    noise = torch.randn_like(curr_sample)
    return torch.sqrt(torch.tensor(1 - beta * beta)) * curr_sample + beta * noise


def pCN_sample(N, burn_in, sample_fn, device, model, start_x, y, H, std_y, beta):

    noisy_posterior = []
    posterior = []

    curr_sample = start_x
    curr_denoised = sample_fn(model, start_x=curr_sample)[0]

    pbar = tqdm(range(N))
    for i in pbar:

        candidate = generate_candidate(curr_sample, beta).to(device)
        candidate_denoised = sample_fn(model, start_x=candidate)[0]   

        accept = acceptance_probability(curr_denoised, candidate_denoised, y, H, std_y)
        rand = torch.rand_like(accept).to(device)

        pbar.set_description(
            (
                f'Iteration: {i}. Acceptance probability: {accept[0]}'
            )
        )

        if rand < accept:
            curr_sample = candidate
            curr_denoised = candidate_denoised

            if i >= burn_in:
                noisy_posterior.append(curr_sample)
                posterior.append(curr_denoised)
    
    return torch.cat(noisy_posterior), torch.cat(posterior)


def subprocess(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    config = dict2namespace(config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using {} device'.format(device))

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

    latents = torch.randn((config.data.num_samples, config.data.dim), device='cpu')

    def scaler(x):
        return x

    sample_fn = get_sampling_fn(
        config, sde, shape=sample_shape, inverse_scaler=scaler, eps=config.sampling.eps, device=device)
    
    start_x = torch.randn(sample_shape, device=device)
    
    N = config.sampling.N
    burn_in = config.sampling.burn_in

    print('Using measurement y = {}'.format(y))

    noisy_posterior, posterior = pCN_sample(N, burn_in, sample_fn, device, model, start_x, 
                                            y, H, std_y, config.sampling.beta)

    basedir = config.log.basedir
    basedir = os.path.join('exp', basedir)
    os.makedirs(basedir, exist_ok=True)
    figsdir = os.path.join(basedir, 'figs')
    os.makedirs(figsdir, exist_ok=True)
    ckptsdir = os.path.join(basedir, 'ckpts')
    os.makedirs(ckptsdir, exist_ok=True)

    np_fig_path = os.path.join(figsdir, 'noisy_posterior.png')
    p_fig_path = os.path.join(figsdir, 'posterior.png')
    fig_path = os.path.join(figsdir, 'post_prior.png')
    true_path = os.path.join(figsdir, 'true_posterior.png')

    np_ckpt_path = os.path.join(ckptsdir, 'noisy_posterior.pt')
    p_ckpt_path = os.path.join(ckptsdir, 'posterior_ckpt.pt')

    true_posterior = compute_posterior(prior, y, H, std_y, num_samples=config.sampling.posterior_samples)

    plot_data([posterior.cpu()], 'Sampled Posterior Distribution\ny: {} H: [{}, {}]'.format(y, H[0], H[1]), 
              p_fig_path)

    plot_data([latents.cpu(), noisy_posterior.cpu()], 'Noisy Posterior Latents (Red) and Diffusion Sampling Prior (Blue)', 
              np_fig_path)

    plot_data([x.cpu(), posterior.cpu()], 'Prior (Blue) + Sampled Posterior (Red)\ny: {} H: [{}, {}]'.format(y, H[0], H[1]), 
              fig_path)
    
    plot_data([x.cpu(), true_posterior.data.cpu(), posterior.cpu()], 
              'True Posterior (Orange) + Sampled Posterior (Green) + Prior (Blue)', true_path)

    torch.save(noisy_posterior, np_ckpt_path)
    torch.save(posterior, p_ckpt_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/gaussian.yml')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ckpt', type=str, default='none')
    args = parser.parse_args()
    subprocess(args)