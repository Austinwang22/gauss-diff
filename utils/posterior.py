import torch
import numpy as np
from utils.datasets import GaussianMixture
from matplotlib import pyplot as plt
from torch.distributions.normal import Normal


def compute_posterior(prior, y, H, std_y, num_samples=1000):
    if len(prior.cov) != 2:
        print('not yet implemented...')
        return
    mean = prior.mean.cpu()
    cov = prior.cov.cpu()
    H = H.cpu()

    new_cov = torch.inverse(torch.matmul(H.T, (1 / std_y) * H) + torch.inverse(cov))
    new_mean = []

    for i in range(len(mean)):
        new_mean.append(torch.matmul(new_cov, H.T * (1 / std_y) * y + torch.matmul(torch.inverse(cov), mean[i])))

    new_mean = torch.stack(new_mean, dim=0)
    weights = compute_weights(y, mean, cov, H, std_y)
    
    posterior = GaussianMixture(num_samples, new_mean, cov, prior=weights)
    return posterior


def compute_weights(y, mean, cov, H, std_y):
    if len(cov) != 2:
        print('not yet implemented...')
        return
    n = len(mean)

    var_y = torch.matmul(H, torch.matmul(cov, H.T)) + std_y ** 2

    denominator = 0
    for i in range(n):
        denominator += (1 / n) * torch.exp(Normal(torch.matmul(H, mean[i]), torch.sqrt(var_y)).log_prob(torch.tensor(y)))

    weights = []
    for i in range(n):
        mean_y = torch.matmul(H, mean[i])
        gauss_pdf = Normal(mean_y, torch.sqrt(var_y))
        numerator = (1 / n) * torch.exp(gauss_pdf.log_prob(torch.tensor(y)))
        weights.append(round((numerator / denominator).item(), 3))
    return weights



mean = torch.tensor([[4.0, 4.0], [-4.0, 4.0], [0., 0.]])
cov = torch.eye(2) * 0.1

mix = GaussianMixture(1000, mean, cov)

H = torch.tensor([1., 1.])

posterior = compute_posterior(mix, 0, H, 0.1)

plt.scatter(posterior.data[:, 0], posterior.data[:, 1])
plt.savefig('test.png')