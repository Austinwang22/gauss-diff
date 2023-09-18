import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns


def plot(ts, samples, xlabel, ylabel, title='', savepath='exp/default/test.png'):
    ts = ts.cpu()
    samples = samples.squeeze().t().cpu()
    plt.figure()
    for i, sample in enumerate(samples):
        plt.plot(ts, sample, label=f'sample {i}')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(prop={'size': 5})
    plt.savefig(savepath)


def plot_data(data, title, save_path):
    plt.figure()
    plt.title(title)
    for i in range(len(data)):
        plt.scatter(data[i].cpu()[:, 0], data[i].cpu()[:, 1])
    plt.savefig(save_path)
    plt.close()


def plot_samples(samples, data_samples, save_path, dim=2):
    if dim == 2:
        x_array = np.concatenate(
            (samples[..., 0], data_samples[..., 0]), axis=None)
        y_array = np.concatenate(
            (samples[..., 1], data_samples[..., 1]), axis=None)
        dat = {'x': x_array,
            'y': y_array,
            'label': ['generated'] * samples.shape[0] + ['data'] * data_samples.shape[0]}

        df = pd.DataFrame(data=dat)
        fig = sns.jointplot(data=df, x='x', y='y', kind='scatter', hue='label')
        fig.savefig(save_path)
    elif dim == 1:
        return


def plot_dict(data_dict, save_path):
    x_list = []
    y_list = []
    label_list = []
    for key, value in data_dict.items():
        x_list.append(value[..., 0])
        y_list.append(value[..., 1])
        label_list += [key] * value.shape[0]

    x_array = np.concatenate(x_list, axis=None)
    y_array = np.concatenate(y_list, axis=None)
    dat = {
        'x': x_array,
        'y': y_array,
        'label': label_list
    }
    df = pd.DataFrame(data=dat)
    fig = sns.jointplot(data=df, x='x', y='y', kind='scatter', hue='label')
    fig.savefig(save_path)
