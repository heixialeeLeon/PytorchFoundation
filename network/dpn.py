import os
import time
import torch
from torch import nn
from torch.utils import data
import torchvision
from pretrainedmodels import models


class DPN68(torch.nn.Module):
    def __init__(self, num_classes, fix_layers=0, dropout_p=None, pretrained=False):
        super(DPN68, self).__init__()
        self.model_name = "DPN68"
        pretrained_model = models.dpn68()
        features = list(pretrained_model.features.children())
        assert fix_layers < len(features)
        for layer in features[:fix_layers+1]:
            for p in layer.parameters():
                p.requires_grad = False
        self.features = nn.Sequential(
            *features
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.classifier = nn.Linear(in_features=832, out_features=num_classes, bias=True)
        nn.init.kaiming_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        
    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.classifier(x)
        return x

    def get_features(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        return x

    def load(self, path, init_fc=False):
        self.load_state_dict(torch.load(path))
        if init_fc:
            nn.init.kaiming_normal_(self.classifier.weight)
            nn.init.constant_(self.classifier.bias, 0)

    def save(self, name=None, prefix='checkpoint/'):
        save_path = os.path.join(prefix, self.model_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if name is None:
            name = time.strftime('%m%d_%H:%M:%S.pth')
        save_model = os.path.join(save_path, name)
        torch.save(self.state_dict(), save_model)
        return save_model


class DPN68b(torch.nn.Module):
    def __init__(self, num_classes, fix_layers=0, dropout_p=None, pretrained=False):
        super(DPN68b, self).__init__()
        self.model_name = "DPN68b"
        pretrained_model = models.dpn68b()
        features = list(pretrained_model.features.children())
        assert fix_layers < len(features)
        for layer in features[:fix_layers+1]:
            for p in layer.parameters():
                p.requires_grad = False
        self.features = nn.Sequential(
            *features
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.classifier = nn.Linear(in_features=832, out_features=num_classes, bias=True)
        nn.init.kaiming_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        
    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.classifier(x)
        return x

    def get_features(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        return x

    def load(self, path, init_fc=False):
        self.load_state_dict(torch.load(path))
        if init_fc:
            nn.init.kaiming_normal_(self.classifier.weight)
            nn.init.constant_(self.classifier.bias, 0)

    def save(self, name=None, prefix='checkpoint/'):
        save_path = os.path.join(prefix, self.model_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if name is None:
            name = time.strftime('%m%d_%H:%M:%S.pth')
        save_model = os.path.join(save_path, name)
        torch.save(self.state_dict(), save_model)
        return save_model


if __name__ == "__main__":
    net = DPN68b(10)
    x = torch.rand((2, 3, 224, 224))
    out = net(x)
    print(net)
    print(x.shape)
    print(out.shape)
