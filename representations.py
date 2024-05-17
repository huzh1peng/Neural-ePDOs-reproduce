import torch
import numpy as np
from math_utils import direct_sum

class diff_rep:
    def __init__(self, n, flip=False):
        self.rep_e = [torch.tensor([[1.]])]
        self.rep_m = [torch.tensor([[1.]])]
        self.order = 1
        self.n = n
        t = 2 * np.pi / n
        self.flip = flip
        self.g_e = torch.tensor([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        if flip:
            self.g_m = torch.tensor([[-1., 0.], [0., 1.]])
            self.rep_m.append(self.g_m)
        self.rep_e.append(self.g_e)

    def next(self):
        if self.order == 4:
            raise ValueError('Order is less than 5')
        a = self.rep_e[self.order]
        self.order += 1
        b = torch.zeros(self.order + 1, self.order + 1)
        for i in range(self.order):
            for j in range(1, self.order):
                b[i, j] = self.g_e[0, 0] * a[i, j] + self.g_e[0, 1] * a[i, j - 1]
            b[i, 0] = self.g_e[0, 0] * a[i, 0]
            b[i, self.order] = self.g_e[0, 1] * a[i, self.order - 1]
        for j in range(1, self.order):
            b[self.order, j] = self.g_e[1, 0] * a[i, j] + self.g_e[1, 1] * a[i, j - 1]
        b[self.order, 0] = self.g_e[1, 0] * a[i, 0]
        b[self.order, self.order] = self.g_e[1, 1] * a[i, self.order - 1]
        self.rep_e.append(b)
        if self.flip:
            a = self.rep_m[self.order - 1]
            b = torch.zeros(self.order + 1, self.order + 1)
            for i in range(self.order):
                for j in range(1, self.order):
                    b[i, j] = self.g_m[0, 0] * a[i, j] + self.g_m[0, 1] * a[i, j - 1]
                b[i, 0] = self.g_m[0, 0] * a[i, 0]
                b[i, self.order] = self.g_m[0, 1] * a[i, self.order - 1]
            for j in range(1, self.order):
                b[self.order, j] = self.g_m[1, 0] * a[i, j] + self.g_m[1, 1] * a[i, j - 1]
            b[self.order, 0] = self.g_m[1, 0] * a[i, 0]
            b[self.order, self.order] = self.g_m[1, 1] * a[i, self.order - 1]
            self.rep_m.append(b)

    def __getitem__(self, i):
        if i > 4:
            raise ValueError('Order is less than 5')
        while i > self.order:
            self.next()
        if self.flip:
            return self.rep_e[i], self.rep_m[i]
        else:
            return self.rep_e[i]

    def e(self, n):
        if self.flip:
            a, _ = self[n]
        else:
            a = self[n]
        if n != 0:
            return direct_sum(self.e(n - 1), a)
        else:
            return a

    def m(self, n):
        if self.flip:
            _, a = self[n]
        else:
            a = self[n]
        if n != 0:
            return direct_sum(self.e(n - 1), a)
        else:
            return a

class c_regular:
    def __init__(self, n):
        self.type = 'cn'
        self.n = n
        self.rep_e = torch.zeros(n, n)
        self.rep_e[1:n, 0:n - 1] = torch.eye(n - 1)
        self.rep_e[0, n - 1] = 1.
        self.dim = n

class d_regular:
    def __init__(self, n):
        self.n = n
        self.type = 'dn'
        self.rep_e = torch.zeros(2 * n, 2 * n)
        self.rep_e[1:n, 0:n - 1] = torch.eye(n - 1)
        self.rep_e[0, n - 1] = 1.
        self.rep_e[n:2 * n - 1, n + 1:2 * n] = torch.eye(n - 1)
        self.rep_e[2 * n - 1, n] = 1.
        self.rep_m = torch.zeros(2 * n, 2 * n)
        self.rep_m[0:n, n:2 * n] = torch.eye(n)
        self.rep_m[n:2 * n, 0:n] = torch.eye(n)
        self.dim = 2 * n

class c_qotient:
    def __init__(self, n, m):
        self.n = n
        self.dim = n // m
        rep = c_regular(n // m)
        self.rep_e = rep.rep_e

class trivial:
    def __init__(self):
        self.dim = 1
        self.type = 'trivial'
        self.rep_e = torch.eye(1)
        self.rep_m = torch.eye(1)
