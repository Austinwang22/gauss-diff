import torch
from torch.utils.data import Dataset
import numpy as np


class Gaussian(Dataset):
    def __init__(self, num_samples, mean, std) -> None:
        super().__init__()
        self.data = mean[None, :] + std[None, :] * \
            torch.randn((num_samples, mean.shape[0]))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]
    

class GaussianMixture(Dataset):
    def __init__(self, num_samples, mean, std, prior=None) -> None:
        '''
        Args:
            num_samples: number of samples to generate
            mean: a list of means of the Gaussian mixture
            std: a list of std of the Gaussian mixture
            prior: prior probability of each mode, if None, uniform prior is used
        '''
        super().__init__()
        self.num_samples = num_samples
        self.data = self.generate(num_samples, mean, std, prior)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples

    @staticmethod
    def generate(num_samples, mean, std, prior):
        # generate samples from a Gaussian mixture given the prior
        # sample latent code
        latents = np.random.choice(len(mean), size=num_samples, p=prior)
        samples = []
        for i in range(len(mean)):
            num_samples_per_mode = np.sum(latents == i)
            sample_mean = mean[i]
            sample_std = std[i] if len(std.shape) == 3 else std
            rand = torch.randn((num_samples_per_mode, sample_mean.shape[0]), device=mean.device) @ sample_std
            sample_per_mode = sample_mean + rand
            samples.append(sample_per_mode)

        samples = torch.cat(samples, dim=0)
        return samples
    