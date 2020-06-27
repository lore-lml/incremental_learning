import torch.nn as nn
import torch
import numpy as np
import math

from torch.nn import Parameter
import torch.nn.functional as F

import copy
"""
Credits to @hshustc
Taken from https://github.com/hshustc/CVPR19_Incremental_Learning/tree/master/cifar100-class-incremental
"""


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100, classifier=None, layer_type='linear', gamma_method='single'):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        if classifier == 'cosine':
            # a downsample layer (conv1x1) is added in the first BasicBlock to adjust
            # input to the output lower dimension caused by stride=2
            self.layer3 = self._make_layer(block, 64, layers[2], stride=2, last_phase=True)
        else:
            self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)

        if classifier == 'pl':
            self.fc = ProgressiveWALayer(64 * block.expansion, num_classes, layer_type=layer_type, gamma_method=gamma_method)
        else:
            self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, last_phase=False):

        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        if last_phase:
            for i in range(1, blocks - 1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, last=True))
        else:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, features=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if features:
            x = x / x.norm(dim=1).unsqueeze(1)
        else:
            x = self.fc(x)

        return x

    def weight_align(self, step_b):
        self.fc.store_and_align_weights_before_classify(step_b)


class ExemplarGenerator(nn.Module):

    def __init__(self, fc, device='cuda'):
        super(ExemplarGenerator, self).__init__()
        self.fc = fc
        self.device = device
        self.mean_std = {}

    def _build_data_dict(self, features, labels):
        data = dict()
        for label, feat in zip(labels, features):
            data[label] = data.get(label, [])
            data[label].append(feat)
        return data

    def _compute_mean_std(self, data, override=True):
        for label, features in data.items():
            if override and label in self.mean_std:
                continue
            features = np.vstack(features)
            mean = features.mean(axis=0)
            std = features.std(axis=0)
            self.mean_std[label] = {'mean': mean, 'std': std}

    def add_data(self, features, labels, override=True):
        data = self._build_data_dict(features, labels)
        self._compute_mean_std(data, override)

    def generate_features(self, labels, n_features):
        import random
        features = []
        label_tensor = []
        for label in labels:
            mu = self.mean_std[label]['mean']
            sigma = self.mean_std[label]['std']
            shape = (n_features, len(mu))
            features.append(np.random.normal(mu, sigma, shape))
            label_tensor.append([label] * n_features)

        label_tensor = np.hstack(label_tensor)
        features = np.vstack(features)
        features = np.array(features, dtype=np.float32)

        return torch.from_numpy(features).to(self.device), torch.from_numpy(np.array(label_tensor)).to(self.device)

    def forward(self, features):
        return self.fc(features)


class ProgressiveWALayer(nn.Module):

    def __init__(self, in_features, out_features, layer_type='linear', gamma_method='single'):
        super(ProgressiveWALayer, self).__init__()
        if gamma_method not in ['single', 'multi']:
            raise ValueError("gamma must be single or multi")

        self.gamma_method = gamma_method
        self.in_features = in_features
        self.out_features = out_features

        if layer_type not in ['linear', 'cosine']:
            raise ValueError("layer_type must be linear or cosine")
        # function to create layers based on layer_type
        self.create_layer = linear_layer if layer_type == 'linear' else cosine_layer
        # classifier used to train net during i_th stage
        self.classifier = self.create_layer(in_features, out_features)

        # Storing memory for the old classifiers
        #self.PL_layers = nn.ModuleList()
        self.weights_per_batch = []

    def forward(self, features):
        return self.classifier(features)

    def store_and_align_weights_before_classify(self, step):
        assert step < 10
        start_batch = step*10
        end_batch = (step+1)*10
        weights = self.classifier.weight.cpu().detach().numpy()[start_batch: end_batch]
        self.weights_per_batch.append(weights)

        if step > 0:
            if self.gamma_method == 'multi':
                self._multi_gamma(step)
            else:
                w_to_norm = np.vstack(self.weights_per_batch[:step])
                norm_old = np.linalg.norm(w_to_norm, axis=1)
                norm_new = np.linalg.norm(weights, axis=1)
                gamma = norm_old.mean() / norm_new.mean()

                old_w = self.classifier.weight.cpu().detach().numpy()
                old_w[:start_batch] = gamma * old_w[:start_batch]
                new_w = Parameter(torch.Tensor(old_w).cuda())

                self.classifier.weight = new_w

    def _multi_gamma(self, step):
        new_weights = self.weights_per_batch[-1]  # the lasts weights of kth batch
        norm_new = np.linalg.norm(new_weights, axis=1)
        mean_new = norm_new.mean()

        new_w = []
        old_w = self.classifier.weight.cpu().detach().numpy()
        for i in range(step):
            w = self.weights_per_batch[i]
            norm_old = np.linalg.norm(w, axis=1)
            gamma = norm_old.mean() / mean_new

            start_batch = i*10
            end_batch = (i+1)*10
            new_w.append(gamma * old_w[start_batch: end_batch])

        new_w = np.vstack(new_w)
        new_w = Parameter(torch.Tensor(new_w).cuda())
        self.classifier.weight[:step*10] = new_w


def cosine_layer(in_features, out_features, sigma=True):
    return CosineLayer(in_features, out_features, sigma=sigma)


def linear_layer(in_features, out_features, bias=False):
    return nn.Linear(in_features, out_features, bias=bias)


class CosineLayer(nn.Module):

    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLayer, self).__init__()

        #Layer dimensions
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))

        # Sigma parameter
        self.sigma = Parameter(torch.Tensor(1)) if sigma else None

        # Reset layer parameter
        self.reset_parameters()

    def reset_parameters(self):

        std = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std, std)

        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):

        # Compute output
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))

        # Scale by sigma if set
        if self.sigma is not None:
            out = self.sigma * out

        return out


def resnet32(device, **kwargs):
    n = 5
    model = ResNet(BasicBlock, [n, n, n], **kwargs)
    generator = ExemplarGenerator(model.fc, device)
    return model, generator


def resnet_progressive_layers(**kwargs):
    n = 5
    return ResNet(BasicBlock, [n, n, n], **kwargs)
