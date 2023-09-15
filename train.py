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
from plots import plot_samples
from matplotlib import pyplot as plt


def loss_fn(model, sde, x, eps=0.0):
    t = torch.rand(x.shape[0], device=x.device) * (sde.T - eps) + eps
    z = torch.randn_like(x)
    mean, std = sde.marginal_prob(x, t)
    perturbed_data = mean + std[:, None] * z
    score = model(perturbed_data, t)
    losses = torch.square(z - score)
    loss = torch.mean(losses)
    return loss


def train_fn(model, trainloader, optimizer, config, device):
    # setup directories
    basedir = config.log.basedir
    basedir = os.path.join('exp', basedir)
    os.makedirs(basedir, exist_ok=True)
    ckpt_dir = os.path.join(basedir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)
    fig_dir = os.path.join(basedir, 'figs')
    os.makedirs(fig_dir, exist_ok=True)
    # plot dataset samples
    data_samples = trainloader.dataset.data.numpy()

    # parse args
    grad_clip = config.optim.grad_clip
    num_epoch = config.train.num_epoch
    save_step = config.train.save_step
    eval_step = config.train.eval_step
    eps = config.sampling.eps
    # setup sde
    if config.model.sde == 'VP':
        sde = VPSDE()
    elif config.model.sde == 'VE':
        sde = VESDE()
    elif config.model.sde == 'EDM':
        sde = EDMSDE()
    print(f'SDE type: {config.model.sde}')
    # setup sampling function
    num_samples = config.sampling.num_samples
    sample_shape = (num_samples, config.data.dim)

    def scaler(x):
        return x

    sample_fn = get_sampling_fn(
        config, sde, shape=sample_shape, inverse_scaler=scaler, eps=eps, device=device)
    # ---------
    pbar = tqdm(range(num_epoch))
    for e in pbar:
        train_loss = 0.0
        for data in trainloader:
            optimizer.zero_grad()
            data = data.to(device)
            loss = loss_fn(model, sde, data, eps=eps)
            loss.backward()
            if grad_clip > 0.:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip)
            optimizer.step()
            train_loss += loss.item()
        avg_loss = train_loss / len(trainloader)
        pbar.set_description(
            (
                f'Epoch: {e}. Train loss: {avg_loss}'
            )
        )
        if e % save_step == 0 and e > 0:
            ckpt_path = os.path.join(ckpt_dir, f'model-{e}.pt')
            # save_ckpt(ckpt_path, model, optim=optimizer)
            torch.save(model.state_dict(), ckpt_path)

        if eval_step > 0 and e % eval_step == 0 and e > 0:
            samples, n = sample_fn(model)
            samples = samples.cpu().numpy()
            fig_path = os.path.join(fig_dir, f'{e}.png')
            print(fig_path)
            # plot_samples(samples, data_samples, fig_path)
            plt.figure()
            plt.scatter(samples[:, 0], samples[:, 1])
            plt.savefig(fig_path)
            plt.close()

    data_path = os.path.join(fig_dir, 'data.png')
    plt.figure()
    plt.scatter(data_samples[:, 0], data_samples[:, 1])
    plt.savefig(data_path)
    plt.close()
            
    print('Finish training')


def subprocess(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    config = dict2namespace(config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using {} device'.format(device))
    # model = MLP(config).to(device)
    model = Network(device).to(device)
    optimizer = Adam(model.parameters(), lr=config.optim.lr)

    mean = torch.tensor(config.data.mean, dtype=torch.float32)
    std = torch.tensor(config.data.std, dtype=torch.float32)
    type = config.data.type
    
    if type == 'gaussian':
        dataset = Gaussian(num_samples=config.data.num_samples,
                        mean=mean, std=std)
    elif type == 'mixture':
        dataset = GaussianMixture(num_samples=config.data.num_samples,
                                mean=mean, std=std)
    trainloader = DataLoader(
        dataset, batch_size=config.train.batchsize, shuffle=True)
    
    ckpt = args.ckpt
    if ckpt != 'none':
        model.load_state_dict(torch.load(ckpt))
        print('Loading preset weights...')

    torch.save(dataset.data, 'exp/mcmc-mixture-3mode-vp/x.pt')

    train_fn(model, trainloader, optimizer, config, device)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/gaussian.yml')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ckpt', type=str, default='none')
    args = parser.parse_args()
    subprocess(args)
