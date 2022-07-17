import tqdm
import torch
import numpy as np
import torch.nn as nn
from model import ClsNet
from data import data_pipeline
from tensorboardX import SummaryWriter
from torch.optim import SGD, Adam, lr_scheduler


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train(epochs, batch_size, lr, data1, data2):
    train_loader, val_loader = data_pipeline(data1, data2, batch_size)
    model = ClsNet(2, 2, 0.2)
    model = model.to(device)

    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.9
    for epoch in range(epochs):
        train_loss, val_loss = 0, 0
        train_acc, val_acc = 0, 0

        model.train()
        for x, y in tqdm.tqdm(train_loader):
            x, y = x.squeeze(0).float().to(device), y.squeeze(0).long().to(device)
            predict = model(x)
            loss = criterion(predict.squeeze(), y.squeeze())
            _, max_index = torch.max(predict, dim=-1)
            train_acc += torch.sum((max_index == y).float()) / y.shape[0]
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        for x, y in tqdm.tqdm(val_loader):
            x, y = x.squeeze(0).float().to(device), y.squeeze(0).long().to(device)
            predict = model(x)
            loss = criterion(predict.squeeze(), y.squeeze())
            _, max_index = torch.max(predict, dim=-1)
            val_acc += torch.sum((max_index == y).float()) / y.shape[0]
            val_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        train_acc = train_acc / len(train_loader)

        val_loss = val_loss / len(val_loader)
        val_acc = val_acc / len(val_loader)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), './saved_models/%.2f.pth' % best_val_acc)

        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/acc', train_acc, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)

        print('Epoch : %03d | Train Loss : %.2f | Train Acc : %.2f | Val Loss : %.2f | Val Acc : %.2f' %
              (epoch, train_loss, train_acc, val_loss, val_acc))


if __name__ == '__main__':
    writer = SummaryWriter(logdir='./tensorboard/eccv/')
    train(epochs=500, batch_size=100, lr=1e-4, data1=[3.5, 1], data2=[-3.5, 1])







