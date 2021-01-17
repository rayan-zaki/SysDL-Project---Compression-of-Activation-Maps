import torch
import torch.nn as nn
from torch import Tensor
import torchvision.models as models
from typing import Callable


class Sparsifier(nn.Module):
    def __init__(self, model: nn.Module, alpha):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self._loss = 0

        for name, module in self.model.named_modules():
            if type(module) != nn.Sequential:
                module.register_forward_hook(self.update_loss_hook())

    def update_loss_hook(self) -> Callable:
        def fn(layer, _, output):
            self._loss += self.alpha[layer] * torch.norm(output, 1, -1)
        return fn

    def forward(self, x: Tensor):
        return self.model(x), self._loss


model = models.vgg16(pretrained=True)

alpha = {}
for name, layer in model.named_modules():
    if type(layer) != nn.Sequential:
        alpha[layer] = 0.001

new_model = Sparsifier(model, alpha)
