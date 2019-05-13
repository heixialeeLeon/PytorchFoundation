import os
import time
import torch
from torch import nn
from torch.utils import data
import torchvision
from torchvision import models


class ResNet18(torch.nn.Module):
    def __init__(self, num_classes, fix_layers=0, dropout_p=None, pretrained=False):
        super(ResNet18, self).__init__()
        self.model_name = "ResNet18"
        pretrained_model = models.resnet18(pretrained=pretrained)
        features = list(pretrained_model.children())[:-2]
        assert fix_layers < len(features)
        for layer in features[:fix_layers+1]:
            for p in layer.parameters():
                p.requires_grad = False
        self.features = nn.Sequential(*features)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.classifier = nn.Linear(in_features=512, out_features=num_classes, bias=True)
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


class ResNet34(torch.nn.Module):
    def __init__(self, num_classes, fix_layers=0, dropout_p=None, pretrained=False):
        super(ResNet34, self).__init__()
        self.model_name = "ResNet34"
        pretrained_model = models.resnet34(pretrained=pretrained)
        features = list(pretrained_model.children())[:-2]
        assert fix_layers < len(features)
        for layer in features[:fix_layers+1]:
            for p in layer.parameters():
                p.requires_grad = False
        self.features = nn.Sequential(*features)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.classifier = nn.Linear(in_features=512, out_features=num_classes, bias=True)
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


class ResNet50(torch.nn.Module):
    def __init__(self, num_classes, fix_layers=0, dropout_p=None, pretrained=False):
        super(ResNet50, self).__init__()
        self.model_name = "ResNet50"
        pretrained_model = models.resnet50(pretrained=pretrained)
        features = list(pretrained_model.children())[:-2]
        assert fix_layers < len(features)
        for layer in features[:fix_layers+1]:
            for p in layer.parameters():
                p.requires_grad = False
        self.features = nn.Sequential(*features)
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


class ResNet101(torch.nn.Module):
    def __init__(self, num_classes, fix_layers=0, dropout_p=None, pretrained=False):
        super(ResNet101, self).__init__()
        self.model_name = "ResNet101"
        pretrained_model = models.resnet101(pretrained=pretrained)
        features = list(pretrained_model.children())[:-2]
        assert fix_layers < len(features)
        for layer in features[:fix_layers+1]:
            for p in layer.parameters():
                p.requires_grad = False
        self.features = nn.Sequential(*features)
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
    model = ResNet18(10)
    print(model)
    