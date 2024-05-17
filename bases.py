import torch
import numpy as np
from math_utils import kronecker, solve

def c_lin_base(in_rep, out_rep):
    A = torch.eye(in_rep.size(0) * out_rep.size(1)) - kronecker(out_rep, torch.inverse(in_rep.transpose(0, 1)))
    w = solve(A).transpose(0, 1).reshape(-1, out_rep.size(0), in_rep.size(0))
    return w

def d_lin_base(type_in, type_out, type_in_m, type_out_m):
    A = torch.eye(type_in.size(0) * type_out.size(1)) - kronecker(type_out, torch.inverse(type_in.transpose(0, 1)))
    A_ = torch.eye(type_in_m.size(0) * type_out_m.size(1)) - kronecker(type_out_m, torch.inverse(type_in_m.transpose(0, 1)))
    w = solve(torch.cat((A, A_), dim=0)).transpose(0, 1).reshape(-1, type_out.size(0), type_in.size(0))
    return w

def cat_lin_base(a, b, dim=0):
    b1, n1, m1, h, w = a.shape
    b2, n2, m2, h, w = b.shape
    if dim == 0:
        c = torch.zeros((b1 + b2, n1 + n2, m1, h, w))
        c[:b1, :n1] = a
        c[b1:, n1:] = b
    else:
        c = torch.zeros((b1 + b2, n1, m1 + m2, h, w))
        c[:b1, :, :m1] = a
        c[b1:, :, m1:] = b
    return c

def kaiming_init(base, num_in, num_out, normal=True):
    f = torch.sum(base * base) * num_in / base.size(1)
    if normal:
        weight = torch.sqrt(1 / f) * torch.randn(num_in, num_out, base.size(0))
    else:
        weight = torch.sqrt(12 / f) * (torch.rand(num_in, num_out, base.size(0)) - 0.5)
    return weight

def make_rbffd(order, kernel_size):
    from rbf.pde.fd import weight_matrix
    diff = []
    coord = make_coord(kernel_size)
    for i in range(order + 1):
        w = weight_matrix(torch.zeros(1, 2).numpy(), coord.numpy(), kernel_size ** 2, [i, order - i],
                          phi='phs6', eps=0.5).toarray()
        w = torch.tensor(w).reshape(kernel_size, kernel_size)
        diff.append(w)
    return torch.stack(diff, 0).to(torch.float32)

def make_gauss(order, kernel_size):
    from rbf.basis import get_rbf
    diff = []
    coord = make_coord(kernel_size)
    gauss = get_rbf('ga')
    for i in range(order + 1):
        w = gauss(coord, torch.zeros(1, 2), eps=0.99, diff=[i, order - i]).reshape(kernel_size, kernel_size)
        diff.append(torch.tensor(w))
    return torch.stack(diff, 0).to(torch.float32)

def make_coord(kernel_size):
    x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1)
    coord = torch.meshgrid([-x, x])
    return torch.stack([coord[1], coord[0]], -1).reshape(kernel_size ** 2, 2)

