import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split


class MyDataset(Dataset):
    def __init__(self, data1, data2, batch_size):
        self.loc1, self.scale1 = data1
        self.loc2, self.scale2 = data2
        self.batch_size = batch_size

    def __len__(self):
        return 10

    def __getitem__(self, index):
        x1 = np.random.normal(self.loc1, self.scale1, size=(self.batch_size, 2))
        x2 = np.random.normal(self.loc2, self.scale2, size=(self.batch_size, 2))
        x = np.concatenate([x1, x2], axis=0)
        y = np.zeros(x.shape[0])
        y[:self.batch_size] = 1
        return x, y


def data_pipeline(data1, data2, batch_size):
    train_set = MyDataset(data1, data2, batch_size)
    val_set = MyDataset(data1, data2, batch_size)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=0)
    return train_loader, val_loader


def gaussian_plt(num):
    x1 = np.random.normal(loc=-3, scale=1, size=(num, 2))
    x2 = np.random.normal(loc=3, scale=1, size=(num, 2))
    plt.scatter(x1[:, 0], x1[:, 1], s=5, c='gray')
    plt.scatter(x2[:, 0], x2[:, 1], s=5, c='gray')
    plt.show()


if __name__ == '__main__':
    data1 = [3, 1]
    data2 = [-3, 1]
    batch_size = 128
    train_loader, val_loader = data_pipeline(data1, data2, batch_size)
    for x, y in train_loader:
        x, y = x.squeeze(0), y.squeeze(0)
        print(x.shape, y.shape, torch.sum(y))