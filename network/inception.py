import os
import time
import torch
from torch import nn
from torch.utils import data
import torchvision
from torchvision import models


class InceptionV3(torch.nn.Module):
    def __init__(self, num_classes, fix_layers=0, dropout_p=None, pretrained=False):
        super(InceptionV3, self).__init__()
        self.model_name = "InceptionV3"
        pretrained_model = models.inception_v3(pretrained=pretrained, aux_logits=True)
        pretrained_features = list(pretrained_model.children())[:-1]
        features = [
            *pretrained_features[0:3],
            nn.MaxPool2d(3, 2),
            *pretrained_features[3:5],
            nn.MaxPool2d(3, 2),
            *pretrained_features[5:13],
            *pretrained_features[14:],
        ]
        assert fix_layers < len(features)
        for layer in features[:fix_layers+1]:
            for p in layer.parameters():
                p.requires_grad = False
        self.features = nn.Sequential(
            *features
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.classifier = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
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
    incep = InceptionV3(10, fix_layers=5)
    x = torch.rand((2, 3, 267, 267))
    out = incep(x)
    print(incep)
    print(x.shape)
    print(out.shape)
