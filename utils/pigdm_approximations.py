import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from models.utils import get_score_fn
from utils.datasets import Gaussian, GaussianMixture
from plots import plot_data
import numpy as np

#Samples from p(x_0 | x_t) = N(hat_x_t, r_t ^2 I), the approximated distribution in Pseudoinverse Guided
#Diffusion Models.
def pigdm_approximation(sde, model, x_t, t, num_samples, device, continuous=True):
    x_t = x_t.unsqueeze(0)
    score_fn = get_score_fn(sde, model, continuous=continuous)
    sigma_t = sde.marginal_prob(x_t, t)[1]
    r_t = torch.sqrt(torch.tensor((sigma_t ** 2) / (sigma_t ** 2 + 1))).to(device)
    vec_t = torch.ones(x_t.shape[0], device=device) * t
    hat_x = x_t + sigma_t * sigma_t * score_fn(x_t, vec_t)
    std = r_t * r_t * torch.eye(x_t.shape[1]).to(device)
    std = std.unsqueeze(0)
    return GaussianMixture(num_samples=num_samples, mean=hat_x.cpu(), std=std.cpu())

#Samples from p(x_0 | x_t) = p(x_t | x_0) p(x_0) / p(x_t), given that p(x_0) is a known Gaussian mixture.
def true_x0_xt_density(sde, prior, x_t, t, num_samples):
    sigma_t = sde.marginal_prob(x_t, t)[1].cpu()
    sigma_0 = sde.marginal_prob(x_t, 0)[1]

    mean = prior.mean.cpu()
    cov = prior.cov.cpu()

    new_cov = torch.inverse(torch.eye(2) / (sigma_t ** 2 - sigma_0 ** 2) + torch.inverse(cov))
    new_mean = []
    
    for i in range(len(mean)):
        new_mean.append(torch.matmul(new_cov, torch.matmul(torch.inverse(cov), mean[i])))

    new_mean = torch.stack(new_mean, dim=0)

    n = len(mean)

    var_x_t = cov + torch.eye(2) * sigma_t ** 2
    var_x_t = torch.maximum(var_x_t, torch.tensor(1e-6)).cpu()

    denominator = 0
    for i in range(n):
        denominator += (1 / n) * torch.exp(MultivariateNormal(mean[i].cpu(), var_x_t).log_prob(x_t.cpu()))
    
    weights = []
    for i in range(n):
        gauss_pdf = MultivariateNormal(mean[i].cpu(), var_x_t)
        numerator = (1 / n) * torch.exp(gauss_pdf.log_prob(x_t.cpu()))
        weights.append((numerator / denominator).item())
    
    weights = np.array(weights)
    weights /= weights.sum()
    print(weights)
    pdf = GaussianMixture(num_samples, new_mean, new_cov, prior=weights)
    return pdf


def compare(sde, model, x_t, t, num_samples, device, prior, approx_path, true_path):
    approx = pigdm_approximation(sde, model, x_t, t, num_samples, device)
    true = true_x0_xt_density(sde, prior, x_t, t, num_samples)
    plot_data([approx.data.detach().cpu()], 'Approximated p(x_0 | x_t)', approx_path)
    plot_data([true.data.detach().cpu()], 'True p(x_0 | x_t)', true_path)