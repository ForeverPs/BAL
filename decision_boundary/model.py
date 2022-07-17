import torch
import torch.nn as nn


class ClsNet(nn.Module):
    def __init__(self, feature_in, num_classes, dropout=0.2):
        super(ClsNet, self).__init__()
        self.feature_in = feature_in
        self.num_classes = num_classes

        self.cls_module = nn.ModuleList([
            self.cls_block(self.feature_in, 128, dropout),
            self.cls_block(128, 64, dropout),
            self.cls_block(64, 32, dropout),
            nn.Linear(32, num_classes)
            # self.cls_block(32, num_classes, dropout, out=True)
        ])

    def cls_block(self, channel_in, channel_out, p, out=False):
        if not out:
            block = nn.Sequential(
                nn.Linear(channel_in, channel_out),
                # nn.BatchNorm1d(channel_out),
                nn.Dropout(p),
                nn.LeakyReLU(0.1),
            )
        else:
            block = nn.Sequential(
                nn.Linear(channel_in, channel_out),
                nn.Softmax(dim=1),
            )
        return block

    def forward(self, x):
        for module_part in self.cls_module:
            x = module_part(x)
        return x


class AECNet(nn.Module):
    def __init__(self, feature_in, num_classes, dropout=0.2):
        super(AECNet, self).__init__()
        self.feature_in = feature_in
        self.num_classes = num_classes

        self.cls_module = nn.ModuleList([
            self.cls_block(self.feature_in, 128, dropout),
            self.cls_block(128, 64, dropout),
            self.cls_block(64, 32, dropout),
            nn.Linear(32, num_classes)
        ])

    def cls_block(self, channel_in, channel_out, p):
        block = nn.Sequential(
            nn.Linear(channel_in, channel_out),
            nn.BatchNorm1d(channel_out),
            nn.Dropout(p),
            nn.LeakyReLU(0.1),
        )
        return block

    def forward(self, x):
        for module_part in self.cls_module:
            x = module_part(x)
        return x




