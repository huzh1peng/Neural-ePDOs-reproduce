import torch
import numpy as np

def test():
    x = torch.randn(10, 16 * 6, 20, 20).cuda()
    from .group import Group
    g = Group(8, True, 'gauss')
    net = g.conv5x5(('regular', 6), ('regular', 6)).cuda()
    rep = g.rep['regular'].rep_e.cuda()
    rep = torch.matmul(rep, rep)
    from .transforms import rot90
    y_ = net(rot90(x, rep))
    y = rot90(net(x), rep)
    print(torch.sum((y - y_) ** 2))
    print(torch.sum(y ** 2) / (10 * 32 * 20 * 20))

def compute_param(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params
