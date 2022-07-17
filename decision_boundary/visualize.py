import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from model import ClsNet, AECNet


def BAL_show():
    d = ClsNet(2, 2, 0.2)
    model_path = 'BAL_models/model.pth'
    d.load_state_dict(torch.load(model_path, map_location='cpu'))
    d = d.eval()

    model = ClsNet(2, 2, 0.2)
    model_path = './saved_models/1.00.pth'
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.eval()

    x = np.linspace(-10, 10, 1000)
    y = np.linspace(-10, 10, 1000)

    X, Y = np.meshgrid(x, y)
    x, y = torch.from_numpy(X).reshape(-1), torch.from_numpy(Y).reshape(-1)
    z = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)
    score1 = nn.Softmax(-1)(model(z.float()))
    score2 = nn.Softmax(-1)(d(z.float()))[:, 1]
    score, _ = torch.max(score1, dim=-1)
    score = (score * score2).detach().numpy().reshape(1000, 1000)

    c = plt.contourf(X, Y, score, 10, alpha=0.95, cmap='hot')
    plt.jet()
    C = plt.contour(X, Y, score, 2, colors='black')
    plt.clabel(C, inline=True, fontsize=10)
    cb = plt.colorbar(c)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    x1 = np.random.normal(loc=3.5, scale=1, size=(500, 2))
    x2 = np.random.normal(loc=-3.5, scale=1, size=(500, 2))

    plt.scatter(x1[:, 0], x1[:, 1], s=3, c='lime')
    plt.scatter(x2[:, 0], x2[:, 1], s=3, c='lime')
    plt.savefig('results/BAL.png', dpi=200)
    # plt.close()
    plt.show()


def AEC_show():
    d = AECNet(2, 2, 0.2)
    model_path = './AEC_models/model.pth'
    d.load_state_dict(torch.load(model_path, map_location='cpu'))
    d = d.eval()

    model = ClsNet(2, 2, 0.2)
    model_path = './saved_models/1.00.pth'
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.eval()

    x = np.linspace(-10, 10, 1000)
    y = np.linspace(-10, 10, 1000)

    X, Y = np.meshgrid(x, y)
    x, y = torch.from_numpy(X).reshape(-1), torch.from_numpy(Y).reshape(-1)
    z = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)
    score1 = nn.Softmax(-1)(model(z.float()))
    score2 = torch.exp(-0.1 * torch.mean(torch.abs(d(z.float()) - z.float()), dim=-1))
    score, _ = torch.max(score1, dim=-1)
    score = (score * score2).detach().numpy().reshape(1000, 1000)

    c = plt.contourf(X, Y, score, 10, alpha=0.95, cmap='hot')
    plt.jet()
    C = plt.contour(X, Y, score, 2, colors='black')
    plt.clabel(C, inline=True, fontsize=10)
    cb = plt.colorbar(c)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    x1 = np.random.normal(loc=3.5, scale=0.5, size=(500, 2))
    x2 = np.random.normal(loc=-3.5, scale=0.5, size=(500, 2))

    plt.scatter(x1[:, 0], x1[:, 1], s=3, c='lime')
    plt.scatter(x2[:, 0], x2[:, 1], s=3, c='lime')
    # plt.savefig('results/AEC.png', dpi=200)
    # plt.close()
    plt.show()


if __name__ == '__main__':
    BAL_show()
    AEC_show()
