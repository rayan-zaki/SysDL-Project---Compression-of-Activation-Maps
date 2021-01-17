import os
import sys
import torch

import torch.nn as nn
import torch.quantization as qn
import torchvision.models as models

from PIL import Image
from torchvision import transforms


class Compressor(nn.Module):
    def __init__(self, model: nn.Module, threshold):
        super().__init__()
        self.model = model
        self.threshold = threshold

        for name, module in self.model.named_modules():
            if type(module) != nn.Sequential:
                module.register_forward_hook(self.compression_hook(name))

    def approximate(self, x):
        if x < self.threshold:
            return '@'
        else:
            return format(x, '.2f')

    def rle(self, x):
        self.size_before_compression = sys.getsizeof(x.storage())
        sh = x.shape
        b = x.view(-1).detach()

        strs = ''
        tt = self.approximate(b[0])
        freq = 1
        for i in range(1, b.shape[0]):
            if self.approximate(b[i]) == tt:
                freq = freq + 1
            else:
                strs = strs + tt
                strs = strs + str(freq) + '#'
                tt = self.approximate(b[i])
                freq = 1
        strs = strs + tt
        strs = strs + str(freq) + '#'

        self.size_after_compression = sys.getsizeof(strs)
        return strs, sh

    def rld(self, encstr, sh):
        i = 0
        dec = 2
        ll = []
        while i < len(encstr):
            is_zero = False
            k = i
            if encstr[k] == '@':
                is_zero = True
                k = k + 1
            else:
                while encstr[k] != '.':
                    k = k + 1
                k = k + dec + 1

            j = k
            while encstr[k] != '#':
                k = k + 1
            freq = int(encstr[j:k])
            for _ in range(freq):
                if is_zero:
                    ll.append(0)
                else:
                    ll.append(float(encstr[i:j]))
            i = k + 1
        tl = torch.tensor(ll)
        tl = tl.view(sh)
        return tl

    def compression_hook(self, layer):
        def fn(module, _, output):
            if output.view(-1).shape[0] <= 100000:
                encstr, sh = self.rle(output)
                output = self.rld(encstr, sh)

                print('Layer ' + layer + ':')
                f = self.size_before_compression
                print('Size before compression: ', f)
                q = self.size_after_compression
                print('Size after compression: ', q)
                print("{0:.2f} times smaller".format(f/q))
                print()
        return fn

    def forward(self, x):
        return self.model(x)


model_fp32 = models.vgg16(pretrained=True)
model_int8 = qn.quantize_dynamic(model_fp32, dtype=torch.qint8)


def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    print("model: ", label, ' \t', 'Size (KB):', size / 1e3)
    os.remove('temp.p')
    return size


# compare the sizes
f = print_size_of_model(model_fp32, "fp32")
q = print_size_of_model(model_int8, "int8")
print("{0:.2f} times smaller".format(f/q))

compressed_model = Compressor(model_int8, threshold=0.1)


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
    ])

img = Image.open("dog.jpg")
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

compressed_model.eval()
compressed_model(batch_t)
