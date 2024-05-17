import torch

def rot90(x, rep):
    b, c, h, w = x.shape
    x = torch.rot90(x, 3, [2, 3])
    x = torch.einsum('bcdhw, ad->bcahw', x.reshape(b, c // rep.size(0), rep.size(0), h, w), rep).reshape(b, c, h, w)
    return x

def reflect(x, rep):
    b, c, h, w = x.shape
    x = x.transpose(2, 3)
    x = torch.rot90(x, 3, [2, 3])
    x = torch.einsum('bcdhw, ad->bcahw', x.reshape(b, c // rep.size(0), rep.size(0), h, w), rep).reshape(b, c, h, w)
    return x
