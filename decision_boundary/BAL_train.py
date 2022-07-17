import tqdm
import torch
import numpy as np
import torch.nn as nn
from model import ClsNet
from data import data_pipeline
from visualize import BAL_show
from tensorboardX import SummaryWriter
from torch.optim import SGD, Adam, lr_scheduler


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train(epochs, batch_size, lr, data1, data2):
    train_loader, val_loader = data_pipeline(data1, data2, batch_size)
    model = ClsNet(2, 2, 0.2)
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_loss, val_loss = 0, 0

        model.train()
        for x, y in tqdm.tqdm(train_loader):
            x = x.squeeze(0).float().to(device)
            y = torch.ones(y.shape).long().to(device)
            predict = model(x)
            loss1 = criterion(predict.squeeze(), y.squeeze())

            x_uni = torch.from_numpy(np.random.uniform(-15, 15, size=(5 * batch_size, 2))).float()
            y_uni = torch.zeros(x_uni.shape[0]).long()
            y_uni_pred = model(x_uni)
            loss2 = criterion(y_uni_pred.squeeze(), y_uni.squeeze())

            loss = loss1 + 0.2 * loss2
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        for x, y in tqdm.tqdm(val_loader):
            x = x.squeeze(0).float().to(device)
            y = torch.ones(y.shape).long().to(device)
            predict = model(x)
            loss1 = criterion(predict.squeeze(), y.squeeze())

            x_uni = torch.from_numpy(np.random.uniform(-10, 10, size=(5 * batch_size, 2))).float()
            y_uni = torch.zeros(x_uni.shape[0]).long()
            y_uni_pred = model(x_uni)
            loss2 = criterion(y_uni_pred.squeeze(), y_uni.squeeze())

            loss = loss1 + 0.2 * loss2
            val_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)

        torch.save(model.state_dict(), 'BAL_models/model.pth')

        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)

        print('Epoch : %03d | Train Loss : %.2f | Val Loss : %.2f' %
              (epoch, train_loss, val_loss))

        BAL_show()


if __name__ == '__main__':
    writer = SummaryWriter(logdir='./tensorboard/eccv/')
    train(epochs=100, batch_size=1000, lr=1e-4, data1=[3.5, 1], data2=[-3.5, 1])







