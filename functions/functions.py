import torch
from torch import nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.nn import Sequential
  
def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)

def nonlinear(inplace=True):
    return nn.ReLU(inplace=inplace)

class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.act = nonlinear()

    def forward(self, t):
        out = self.conv(t)
        out = self.act(out)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_, mid, out, is_deconv=False):
        super().__init__()
        self.in_ = in_
        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_, mid),
                nn.ConvTranspose2d(mid, out, kernel_size=4,stride=2,padding=1),
                nonlinear()
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                ConvRelu(in_, mid),
                ConvRelu(mid, out)
            )

    def forward(self, t):
        return self.block(t)


