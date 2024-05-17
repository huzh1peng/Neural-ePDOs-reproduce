import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_filter(filt_size):
    """
    Return filter coefficients based on the filter size.
    """
    if filt_size == 1:
        a = np.array([1.])
    elif filt_size == 2:
        a = np.array([1., 1.])
    elif filt_size == 3:
        a = np.array([1., 2., 1.])
    elif filt_size == 4:
        a = np.array([1., 3., 3., 1.])
    elif filt_size == 5:
        a = np.array([1., 4., 6., 4., 1.])
    elif filt_size == 6:
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif filt_size == 7:
        a = np.array([1., 6., 15., 20., 15., 6., 1.])
    else:
        raise ValueError(f'Unsupported filter size: {filt_size}')
    return a

def get_pad_layer(pad_type: str):
    """
    Return the appropriate padding layer based on the padding type.
    """
    if pad_type in ['refl', 'reflect']:
        return nn.ReflectionPad2d
    elif pad_type in ['repl', 'replicate']:
        return nn.ReplicationPad2d
    elif pad_type == 'zero':
        return nn.ZeroPad2d
    else:
        raise ValueError(f'Pad type [{pad_type}] not recognized')

def get_pad_layer_1d(pad_type: str):
    """
    Return the appropriate 1D padding layer based on the padding type.
    """
    if pad_type in ['refl', 'reflect']:
        return nn.ReflectionPad1d
    elif pad_type in ['repl', 'replicate']:
        return nn.ReplicationPad1d
    elif pad_type == 'zero':
        return nn.ZeroPad1d
    else:
        raise ValueError(f'Pad type [{pad_type}] not recognized')

class Downsample(nn.Module):
    def __init__(self, pad_type: str = 'reflect', filt_size: int = 3, stride: int = 2, channels: int = None, pad_off: int = 0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        # Calculate padding sizes for each side of the image
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))] * 2
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.channels = channels

        # Get the filter coefficients and normalize
        a = get_filter(filt_size)
        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        # Register filter as buffer to avoid updating it during training
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        # Initialize padding layer
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # Handle the special case where filter size is 1
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

class Downsample1D(nn.Module):
    def __init__(self, pad_type: str = 'reflect', filt_size: int = 3, stride: int = 2, channels: int = None, pad_off: int = 0):
        super(Downsample1D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        # Calculate padding sizes for each side of the signal
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.channels = channels

        # Get the filter coefficients and normalize
        a = get_filter(filt_size)
        filt = torch.Tensor(a)
        filt = filt / torch.sum(filt)
        # Register filter as buffer to avoid updating it during training
        self.register_buffer('filt', filt[None, None, :].repeat((self.channels, 1, 1)))

        # Initialize padding layer
        self.pad = get_pad_layer_1d(pad_type)(self.pad_sizes)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # Handle the special case where filter size is 1
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride]
        else:
            return F.conv1d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])
