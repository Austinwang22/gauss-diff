import torch
import torch.nn as nn


def get_act(name):
    """Get activation functions from the config file."""

    if name == 'elu':
        return nn.ELU()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'lrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    elif name == 'swish':
        return nn.SiLU()
    elif name == 'softplus':
        return nn.Softplus()
    else:
        raise NotImplementedError('activation function does not exist!')


def get_score_fn(sde, model, train=False, continuous=False):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.
    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A score model.
      continuous: If `True`, the score-based model is expected to directly take continuous time steps.
    Returns:
      A score function.
    """

    def score_fn(x, t):
        # Scale neural network output by standard deviation and flip sign
        if continuous:
            # For VP-trained models, t=0 corresponds to the lowest noise level
            # The maximum value of time embedding is assumed to 999 for
            # continuously-trained models.
            labels = t
            score = model(x, labels)
            std = sde.marginal_prob(torch.zeros_like(x), t)[1]
        else:
            # For VP-trained models, t=0 corresponds to the lowest noise level
            labels = t
            score = model(x, labels)
            std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]
        score = -score / std[:, None]
        return score

    return score_fn
