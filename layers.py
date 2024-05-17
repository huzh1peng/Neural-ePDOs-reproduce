import torch
import torch.nn as nn
import math

from math_utils import ortho_basis, kronecker
from bases import kaiming_init, c_lin_base, d_lin_base, cat_lin_base


class conv(nn.Module):
    def __init__(self, base, num_in, num_out, groups=1, stride=1):
        super(conv, self).__init__()
        self.base = torch.nn.Parameter(base, requires_grad=False)
        self.num_in = num_in // groups
        self.num_out = num_out
        self.groups = groups
        self.dim_rep_in = self.base.size(2)
        self.dim_rep_out = self.base.size(1)
        self.bases = self.base.size(0)
        self.param = torch.nn.Parameter(kaiming_init(base, self.num_in, num_out))
        self.stride = stride
        self.size = self.base.size(4)
        if self.stride != 1:
            self.pool = nn.MaxPool2d(self.stride, self.stride)
        self.eval()

    def forward(self, x):
        if self.training:
            self.kernel = torch.matmul(self.param.reshape(-1, self.bases), self.base.reshape(self.bases, -1)).reshape(
                self.num_in, self.num_out, self.dim_rep_out, self.dim_rep_in, self.size, self.size)
            self.kernel = self.kernel.permute((1, 2, 0, 3, 4, 5)).reshape(self.num_out * self.dim_rep_out,
                                                                          self.num_in * self.dim_rep_in, self.size,
                                                                          self.size)
        out = nn.functional.conv2d(x, self.kernel, bias=None, stride=1, padding=math.floor(self.size / 2),
                                   groups=self.groups)
        if self.stride != 1:
            return self.pool(out)
        else:
            return out

    def eval(self):
        self.kernel = torch.einsum('ijk,kmnpq->jminpq', self.param, self.base) \
            .reshape(self.num_out * self.dim_rep_out, self.num_in * self.dim_rep_in, self.size, self.size)
        self.kernel = self.kernel.detach()

class gnorm(nn.Module):
    def __init__(self, in_type, groups):
        super(gnorm, self).__init__()
        self.bn = nn.GroupNorm(groups, in_type[1])
        self.num_in = in_type[1]
        self.weight = nn.Parameter(torch.ones(1, self.num_in, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, self.num_in, 1, 1, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        return (self.bn(x.reshape(b, self.num_in, c // self.num_in, h, w)) * self.weight + self.bias).reshape(x.size())

class GroupBatchNorm(nn.Module):
    def __init__(self, num_rep, dim_rep, affine=False, momentum=0.1, track_running_stats=True):
        super(GroupBatchNorm, self).__init__()
        self.momentum = momentum
        self.bn = nn.BatchNorm3d(num_rep, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.num_rep = num_rep
        self.dim_rep = dim_rep

    def forward(self, x):
        shape = x.shape
        x = self.bn(x.reshape(x.size(0), self.num_rep, self.dim_rep, x.size(2), x.size(3)))
        x = x.reshape(shape)
        return x

class GroupPooling(nn.Module):
    def __init__(self, dim_rep, num_rep, type='max'):
        super(GroupPooling, self).__init__()
        if type == 'avg':
            self.pool = nn.AdaptiveAvgPool3d((dim_rep, 1, 1))
        else:
            self.pool = nn.MaxPool3d((dim_rep, 1, 1))
        self.dim = dim_rep
        self.num_rep = num_rep

    def forward(self, x):
        size = x.size()
        x = x.reshape(x.size(0), self.num_rep, self.dim, x.size(2), x.size(3))
        x = self.pool(x)
        return x.reshape(x.size(0), x.size(1), x.size(3), x.size(4))

class FlipRestrict(nn.Module):
    def __init__(self, n, num_in_rep):
        super(FlipRestrict, self).__init__()
        a = torch.zeros(2 * n, 2 * n)
        a[:n, :n] = torch.eye(n)
        for i in range(n):
            a[n + i, 2 * n - i - 1] = 1.
        self.param = torch.nn.Parameter(a, False)
        self.num_in_rep = num_in_rep
        self.dim = n * 2

    def forward(self, x):
        x = torch.einsum('ij,bkjmn->bkimn', self.param,
                         x.reshape(x.size(0), self.num_in_rep, self.dim, x.size(2), x.size(3))).reshape(x.shape)
        return x

class nlpdo_torch(nn.Module):
    def __init__(self, group, in_type, out_type, order, reduction, s, g, stride=1):
        super(nlpdo_torch, self).__init__()
        self.group = group
        self.reduction = reduction
        self.g = g
        self.stride = stride
        self.rep, self.num_in = in_type
        self.num_mid = int(self.num_in * self.group.dim_rep(self.rep) // (self.reduction * self.group.dim))
        self.order = order
        self.s = s

        self.conv1 = self.group.conv1x1(in_type, ('regular', self.num_mid))
        self.gn = gnorm(('regular', self.num_mid), self.num_mid)

        self.conv2 = self.make_conv2()
        self.conv3 = self.group.conv1x1(in_type, out_type)

    def forward(self, x):
        x_ = nn.functional.relu(self.gn(self.conv1(x)))
        W = self.conv2(x_) / 5
        b, c, h, w = W.shape
        W = W.reshape(b, 1, -1, self.group.dim_rep(self.rep) * 25, h, w)
        W = W / (torch.norm(W, 1, dim=3, keepdim=True) + 0.01)
        W = W.reshape(b, 1, -1, 25, h, w)
        x = nn.functional.unfold(x, 5, padding=2).reshape(b, -1, self.g * self.group.dim_rep(self.rep), 25, h, w)
        out = torch.sum(W * x, dim=3)

        return self.conv3(out.reshape(b, -1, h, w))

    def make_conv2(self):
        self.trans = self.group.trans(self.order).reshape(-1, 25)
        base_ = True
        if isinstance(self.rep, tuple):
            for i in self.rep:
                out_e = kronecker(self.group.rep[i[0]].rep_e,
                                  torch.inverse(self.group.diff_rep.e(self.order)).transpose(0, 1))
                if self.group.flip:
                    out_m = kronecker(self.group.rep[i[0]].rep_m,
                                      torch.inverse(self.group.diff_rep.m(self.order)).transpose(0, 1))
                    base = d_lin_base(self.group.rep['regular'].rep_e, out_e, self.group.rep['regular'].rep_m, out_m)
                else:
                    base = c_lin_base(self.group.rep['regular'].rep_e, out_e)
                n, p, q = base.size()
                base = torch.cat(i[1] * [base], dim=1)
                base = torch.einsum('nklq, lp->nkpq', base.reshape(n, -1, (self.order + 1) * (self.order + 2) // 2, q),
                                    self.trans).reshape(n, -1)
                base = ortho_basis(base).reshape(n, -1, q, 1, 1)
                if base_ is True:
                    base_ = base
                else:
                    base_ = cat_lin_base(base_, base, dim=0)
            return conv(base_, self.num_mid, self.g, groups=self.s)
        else:
            for i in range(self.order + 1):
                if self.group.flip:
                    out_e = kronecker(self.group.rep[self.rep].rep_e,
                                      torch.inverse(self.group.diff_rep[i][0]).transpose(0, 1))
                    out_m = kronecker(self.group.rep[self.rep].rep_m,
                                      torch.inverse(self.group.diff_rep[i][1]).transpose(0, 1))
                    base = d_lin_base(self.group.rep[self.rep].rep_e, out_e, self.group.rep[self.rep].rep_m, out_m)
                else:
                    out_e = kronecker(self.group.rep[self.rep].rep_e,
                                      torch.inverse(self.group.diff_rep[i]).transpose(0, 1))
                    base = c_lin_base(self.group.rep[self.rep].rep_e, out_e)
                if base_ is True:
                    base_ = base.reshape(base.size(0), -1, i + 1, base.size(2), 1)
                else:
                    base_ = cat_lin_base(base_, base.reshape(base.size(0), -1, i + 1, base.size(2), 1), dim=1)
            base = base_.squeeze(-1)
            n, p, _, q = base.size()
            base = torch.einsum('nklq, lp->nkpq', base.reshape(n, -1, (self.order + 1) * (self.order + 2) // 2, q),
                                self.trans).reshape(n, -1)
            base = ortho_basis(base).reshape(n, -1, q, 1, 1)
        return conv(base, self.num_mid, self.g, groups=self.s)
