import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Pre-trained models taken from https://github.com/huyvnphan/PyTorch_CIFAR10.
from cifar10_models.vgg import vgg16_bn
from cifar10_models.resnet import resnet34
from cifar10_models.mobilenetv2 import mobilenet_v2
from torch.utils.data import DataLoader, Subset
from torchvision import datasets as D, transforms as T


# No regularisation by default (empty `alpha`).
class Sparsifier(nn.Module):
    def __init__(self, model: nn.Module, threshold, alpha={}):
        super().__init__()
        self.model = model
        self.threshold = threshold
        self.alpha = alpha
        self.__loss__ = 0

        for name, module in self.model.named_modules():
            if type(module) != nn.Sequential:
                module.register_forward_hook(self.compression_hook(name))

    def compression_hook(self, _):
        def fn(module, _, output):
            output[torch.abs(output) < self.threshold] = 0
            # if self.alpha[module] is not None:
            #     self.__loss__ += self.alpha[module] * torch.norm(output, 1, -1)
        return fn

    def forward(self, x):
        return self.model(x)    #, self.__loss__


transform = T.Compose([
    T.ToTensor(),
    T.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
        )
    ])

testsize = 1000
dataset = D.CIFAR10("./data", train=False, transform=transform, download=True)
dataset = Subset(dataset, range(testsize))
dataloader = DataLoader(dataset, batch_size=2)


def accuracy(model):
    model.eval()
    correct_preds = 0.0
    for data, target in dataloader:
        output = model(data)
        _, predictions = torch.max(F.softmax(output, dim=1), 1)
        correct_preds += torch.sum(predictions == target.data)
    acc = correct_preds / testsize
    return acc.item()


# VGG16
model = vgg16_bn(pretrained=True)

plot_data = []
for threshold in [1e-4, 1e-3, 1e-2, 1e-1]:
    compressed_model = Sparsifier(model, threshold=threshold)
    acc = accuracy(compressed_model)
    plot_data.append([threshold, acc])
plot_df = pd.DataFrame(plot_data, columns=['Threshold', 'Accuracy'])
plot_df.to_csv('./results/vgg16', index=False)
figure = plot_df.plot(x='Threshold', y='Accuracy', logx=True).get_figure()
figure.savefig('./figs/vgg16')

# Resnet34
model = resnet34(pretrained=True)

plot_data = []
for threshold in [1e-4, 1e-3, 1e-2, 1e-1]:
    compressed_model = Sparsifier(model, threshold=threshold)
    acc = accuracy(compressed_model)
    plot_data.append([threshold, acc])
plot_df = pd.DataFrame(plot_data, columns=['Threshold', 'Accuracy'])
plot_df.to_csv('./results/resnet34', index=False)
figure = plot_df.plot(x='Threshold', y='Accuracy', logx=True).get_figure()
figure.savefig('./figs/resnet34')

# MobilenetV2
model = mobilenet_v2(pretrained=True)

plot_data = []
for threshold in [1e-4, 1e-3, 1e-2, 1e-1]:
    compressed_model = Sparsifier(model, threshold=threshold)
    acc = accuracy(compressed_model)
    plot_data.append([threshold, acc])
plot_df = pd.DataFrame(plot_data, columns=['Threshold', 'Accuracy'])
plot_df.to_csv('./results/mobilenet_v2', index=False)
figure = plot_df.plot(x='Threshold', y='Accuracy', logx=True).get_figure()
figure.savefig('./figs/mobilenet_v2')
