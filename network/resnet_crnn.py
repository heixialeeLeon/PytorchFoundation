import torch
import torch.nn as nn
import math

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, batch_first=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)
        N, T, h = recurrent.size()
        t_rec = recurrent.contiguous().view(N * T, h)

        output = self.embedding(t_rec)
        output = output.view(N, T, -1)
        return output


class BidirectionalGRU(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalGRU, self).__init__()

        self.rnn = nn.GRU(nIn, nHidden, bidirectional=True, batch_first=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)
        N, T, h = recurrent.size()
        t_rec = recurrent.contiguous().view(N * T, h)

        output = self.embedding(t_rec)
        output = output.view(N, T, -1)
        return output


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, groups=4, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or inplanes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )
        self.stride = stride

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        residual = self.shortcut(out) if hasattr(self, 'shortcut') else x

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, hiddens, rnn=None, lstm_hidden=256, num_classes=1000):
        super(ResNet, self).__init__()

        self.inplanes = 32
        conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        bn1 = nn.BatchNorm2d(self.inplanes)
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(self.inplanes, hiddens[0], kernel_size=3, stride=1, padding=1,
                               bias=False)
        
        self.inplanes = hiddens[0]
        layers_ = [conv1, bn1, relu, conv2]
        for i in range(len(layers)):
            if i == 0:
                layers_.append(self._make_layer(block, hiddens[i], layers[i]))
            else:
                layers_.append(self._make_layer(block, hiddens[i], layers[i], stride=2))
        self.layers = nn.Sequential(*layers_)

        if rnn is None:
            self.embedding = nn.Linear(hiddens[-1] * block.expansion, num_classes)
        elif rnn == "LSTM":
            self.rnn = BidirectionalLSTM(hiddens[-1] * block.expansion, lstm_hidden, num_classes)
        elif rnn == "GRU":
            self.rnn = BidirectionalGRU(hiddens[-1] * block.expansion, lstm_hidden, num_classes)
        else:
            print("Ignore unsupported rnn type {}, rnn will not be used.".format(rnn))
            self.embedding = nn.Linear(hiddens[-1] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        w = x.size(3)
        out = nn.functional.adaptive_avg_pool2d(x, (1, w))
        out = out.squeeze(2).permute(0, 2, 1).contiguous()

        if hasattr(self, 'rnn'):
            out = self.rnn(out)
        else:
            N, T, C = out.size()
            out = self.embedding(out.contiguous().view(N * T, C)).view(N, T, -1)

        return out


def resnet20_gru(num_classes):
    model = ResNet(Bottleneck, [2, 2, 2], [64, 128, 256], rnn="GRU", lstm_hidden=256, num_classes=num_classes)
    return model

def resnet20_lstm(num_classes):
    model = ResNet(Bottleneck, [2, 2, 2], [64, 128, 256], rnn="LSTM", lstm_hidden=256, num_classes=num_classes)
    return model

def resnet29_gru(num_classes):
    model = ResNet(Bottleneck, [3, 3, 3], [64, 128, 256], rnn="GRU", lstm_hidden=256, num_classes=num_classes)
    return model

def resnet29_lstm(num_classes):
    model = ResNet(Bottleneck, [3, 3, 3], [64, 128, 256], rnn="LSTM", lstm_hidden=256, num_classes=num_classes)
    return model


if __name__ == '__main__':
    input = torch.randn(4, 3, 48, 640)
    model = ResNet(Bottleneck, [3, 3, 2], [64, 128, 256], rnn="GRU", lstm_hidden=256, num_classes=8167)
    model.eval()
    output = model(input)
    print(model)
    print(output.size())
    torch.save(model.state_dict(), 'temp.pth')
