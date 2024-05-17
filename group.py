import torch
from representations import diff_rep, c_regular, d_regular, c_qotient, trivial
from layers import conv, gnorm, GroupBatchNorm, GroupPooling, FlipRestrict, nlpdo_torch
from math_utils import kronecker, solve, ortho_basis
from bases import c_lin_base, d_lin_base, cat_lin_base, make_gauss, make_rbffd
import torch.nn as nn
from Downsample import Downsample

class Group:
    def __init__(self, n, flip=False, dis='fd'):
        self.dif_ker = {
            '0': torch.tensor([[[1.]]]),
            '1': torch.tensor([[[0., 0., 0.], [-0.5, 0., 0.5], [0., 0., 0.]], [[0., 0.5, 0.], [0., 0., 0.], [0., -0.5, 0.]]]),
            '2': torch.tensor([[[0., 0., 0.], [1., -2., 1.], [0., 0., 0.]], [[-0.25, 0, 0.25], [0., 0., 0.], [0.25, 0., -0.25]], [[0., 1., 0.], [0., -2., 0.], [0., 1., 0.]]]),
            '3': torch.tensor([[[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [-0.5, 1., 0., -1., 0.5], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]], [[0., 0., 0., 0., 0.], [0., 0.5, -1., 0.5, 0.], [0., 0., 0., 0., 0.], [0., -0.5, 1., -0.5, 0.], [0., 0., 0., 0., 0.]], [[0., 0., 0., 0., 0.], [0., -0.5, 0., 0.5, 0.], [0., 1., 0., -1., 0.], [0., -0.5, 0., 0.5, 0.], [0., 0., 0., 0., 0.]], [[0., 0., 0.5, 0., 0.], [0., 0., -1., 0., 0.], [0., 0., 0., 0., 0.], [0., 0., 1., 0., 0.], [0., 0., -0.5, 0., 0.]]]),
            '4': torch.tensor([[[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [1., -4., 6., -4., 1.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]], [[0., 0., 0., 0., 0.], [-0.25, 0.5, 0., -0.5, 0.25], [0., 0., 0., 0., 0.], [0.25, -0.5, 0., 0.5, -0.25], [0., 0., 0., 0., 0.]], [[0., 0., 0., 0., 0.], [0., 1., -2., 1., 0.], [0., -2., 4., -2., 0.], [0., 1., -2., 1., 0.], [0., 0., 0., 0., 0.]], [[0., -0.25, 0., 0.25, 0.], [0., 0.5, 0., -0.5, 0.], [0., 0., 0., 0., 0.], [0., -0.5, 0., 0.5, 0.], [0., 0.25, 0., -0.25, 0.]], [[0., 0., 1., 0., 0.], [0., 0., -4., 0., 0.], [0., 0., 6., 0., 0.], [0., 0., -4., 0., 0.], [0., 0., 1., 0., 0.]]]),
            '5': torch.tensor([[[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [0., 0., 1., 0., 0.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]]]),
            '6': torch.tensor([[[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [0., -0.5, 0., 0.5, 0.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]], [[0., 0., 0., 0., 0.], [0., 0., 0.5, 0., 0.], [0., 0., 0., 0., 0.], [0., 0., -0.5, 0., 0.], [0., 0., 0., 0., 0.]]]),
            '7': torch.tensor([[[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [0., 1., -2., 1., 0.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]], [[0., 0., 0., 0., 0.], [0., -0.25, 0, 0.25, 0.], [0., 0., 0., 0., 0.], [0., 0.25, 0., -0.25, 0.], [0., 0., 0., 0., 0.]], [[0., 0., 0., 0., 0.], [0., 0., 1., 0., 0.], [0., 0., -2., 0., 0.], [0., 0., 1., 0., 0.], [0., 0., 0., 0., 0.]]])
        }
        self.filter = {'5': torch.tensor([[1.]]), '6': torch.tensor([[1., 0.], [0., 1.]])}
        self.filter['7'] = torch.eye(3).float()
        self.filter['3'] = torch.eye(4).float()
        self.filter['3'][0, 0] = 0.
        self.filter['3'][3, 3] = 0.
        self.filter['4'] = torch.zeros(5, 5).float()
        self.filter['4'][2, 2] = 1
        self.dis = dis
        self.discritization(self.dis)
        self.n = n
        self.flip = flip
        self.dim = n
        if flip:
            self.dim *= 2
        self.diff_rep = diff_rep(n, flip)
        self.rep = {'regular': None, 'trivial': trivial()}
        if flip:
            self.rep['regular'] = d_regular(n)
        else:
            self.rep['regular'] = c_regular(n)
            for i in range(1, n):
                j = 2 ** i
                if j >= n:
                    break
                self.rep[f'quo_{j}'] = c_qotient(n, j)
        self.bases = [{} for _ in range(8)]
        self.base3 = [{('regular', 'regular'): None, ('regular', 'trivial'): None, ('trivial', 'trivial'): None, ('trivial', 'regular'): None} for _ in range(8)]
        self.fast_base = {}
        self.lin_base = {'ok': None}
        self.base_3x3 = {('regular', 'regular'): None, ('regular', 'trivial'): None, ('trivial', 'trivial'): None, ('trivial', 'regular'): None}

    def discritization(self, dis):
        if dis == 'fd':
            return 0
        elif dis == 'gauss':
            for i in range(5):
                n = i + 5 if i < 3 else i
                self.dif_ker[str(n)] = make_gauss(i, 5)
        else:
            for i in range(5):
                n = i + 5 if i < 3 else i
                self.dif_ker[str(n)] = make_rbffd(i, 5)

    def coef(self, in_rep, out_rep, df):
        d = kronecker(df.transpose(0, 1), in_rep.transpose(0, 1))
        n1 = d.size(0)
        n2 = out_rep.size(0)
        return kronecker(out_rep, torch.eye(n1)) - kronecker(torch.eye(n2), d)

    def base(self, order, in_rep, out_rep):
        if (in_rep, out_rep) in self.bases[order]:
            return self.bases[order][(in_rep, out_rep)]
        in_rep_ = self.rep[in_rep]
        out_rep_ = self.rep[out_rep]
        if self.flip:
            df1, df2 = self.diff_rep[order % 5]
            w1 = self.coef(in_rep_.rep_e, out_rep_.rep_e, df1)
            w2 = self.coef(in_rep_.rep_m, out_rep_.rep_m, df2)
            w = torch.cat((w1, w2))
        else:
            df = self.diff_rep[order % 5]
            w = self.coef(in_rep_.rep_e, out_rep_.rep_e, df)
        w = solve(w).transpose(0, 1)
        dim_in_rep = in_rep_.dim
        dim_out_rep = out_rep_.dim
        n = w.size(0)
        w = w.reshape(n, dim_out_rep, order % 5 + 1, dim_in_rep).transpose(2, 3)
        w = w.float()
        print("shape of w:{}".format(w.shape))
        print("shape of difkernel:{}".format(self.dif_ker[str(order)].shape))
        self.bases[order][(in_rep, out_rep)] = torch.einsum('ijkl,lmn->ijkmn', w, self.dif_ker[str(order)])
        shape = self.bases[order][(in_rep, out_rep)].shape
        if torch.sum(self.bases[order][(in_rep, out_rep)] ** 2) > 0:
            b = self.bases[order][(in_rep, out_rep)].reshape(shape[0], -1)
            b = b / torch.norm(b, dim=1, keepdim=True)
            self.bases[order][(in_rep, out_rep)] = b.reshape(shape)
        return self.bases[order][(in_rep, out_rep)]

    def fast_base_(self, in_rep, out_rep, order):
        orderlist = range(order + 1)
        if (in_rep, out_rep) not in self.fast_base:
            base = [self.base(i + 5 if i < 3 else i, in_rep, out_rep) for i in orderlist]
            base = torch.cat(base)
            shape = base.shape
            self.fast_base[(in_rep, out_rep)] = base
        else:
            base = self.fast_base[(in_rep, out_rep)]
        return base

    def conv5x5(self, in_type, out_type, order=4, stride=1, groups=1):
        orderlist = range(order + 1)
        in_rep, num_in = in_type
        out_rep, num_out = out_type
        if (in_rep, out_rep) in self.fast_base:
            base = self.fast_base[(in_rep, out_rep)]
        else:
            if isinstance(in_rep, tuple) or isinstance(out_rep, tuple):
                if not isinstance(in_rep, tuple):
                    in_rep = ((in_rep, 1),)
                if not isinstance(out_rep, tuple):
                    out_rep = ((out_rep, 1),)
                in_re = sum((i[1] * (i[0],) for i in in_rep), ())
                out_re = sum((i[1] * (i[0],) for i in out_rep), ())
                base = True
                for j in out_re:
                    b = True
                    for i in in_re:
                        if b is True:
                            b = self.fast_base_(i, j, order)
                        else:
                            b = cat_lin_base(b, self.fast_base_(i, j, order), dim=1)
                    if base is True:
                        base = b
                    else:
                        base = cat_lin_base(base, b, dim=0)
                self.fast_base[(in_rep, out_rep)] = base
            else:
                base = self.fast_base_(in_rep, out_rep, order)
        return conv(base, num_in, num_out, groups, stride)

    def lin_base_(self, in_rep, out_rep):
        if (in_rep, out_rep) not in self.lin_base:
            if self.flip:
                base = d_lin_base(self.rep[in_rep].rep_e, self.rep[out_rep].rep_e, self.rep[in_rep].rep_m, self.rep[out_rep].rep_m)
            else:
                base = c_lin_base(self.rep[in_rep].rep_e, self.rep[out_rep].rep_e)
            base = base.unsqueeze(dim=-1).unsqueeze(dim=-1)
            self.lin_base[(in_rep, out_rep)] = base
        return self.lin_base[(in_rep, out_rep)]

    def conv1x1(self, in_type, out_type, stride=1, groups=1):
        in_rep, num_in = in_type
        out_rep, num_out = out_type
        if (in_rep, out_rep) in self.lin_base:
            return conv(self.lin_base[(in_rep, out_rep)], num_in, num_out, groups, stride)
        if isinstance(in_rep, tuple) or isinstance(out_rep, tuple):
            if not isinstance(in_rep, tuple):
                in_rep = ((in_rep, 1),)
            if not isinstance(out_rep, tuple):
                out_rep = ((out_rep, 1),)
            in_re = sum((i[1] * (i[0],) for i in in_rep), ())
            out_re = sum((i[1] * (i[0],) for i in out_rep), ())
            a = True
            for j in out_re:
                b = True
                for i in in_re:
                    if b is True:
                        b = self.lin_base_(i, j)
                    else:
                        b = cat_lin_base(b, self.lin_base_(i, j), dim=1)
                if a is True:
                    a = b
                else:
                    a = cat_lin_base(a, b, dim=0)
            self.lin_base[(in_rep, out_rep)] = a
            return conv(a, num_in, num_out, groups, stride)
        else:
            base = self.lin_base_(in_rep, out_rep)
            return conv(self.lin_base[(in_rep, out_rep)], num_in, num_out, groups, stride)

    def conv1x1_c(self, num_in, in_rep, num_out, out_rep, stride=1, groups=1):
        base = c_lin_base(in_rep, out_rep)
        base = base.unsqueeze(dim=-1).unsqueeze(dim=-1)
        return conv(base, num_in, num_out, groups, stride)

    def conv1x1_d(self, num_in, in_rep_e, in_rep_m, num_out, out_rep_e, out_rep_m, stride=1, groups=1):
        base = d_lin_base(in_rep_e, out_rep_e, in_rep_m, out_rep_m)
        base = base.unsqueeze(dim=-1).unsqueeze(dim=-1)
        return conv(base, num_in, num_out, groups, stride)

    def flip_restrict(self, in_type):
        in_rep, num_in = in_type
        return FlipRestrict(self.n, num_in)

    def dim_rep(self, rep):
        if isinstance(rep, tuple):
            return sum(i[1] * self.rep[i[0]].dim for i in rep)
        return self.rep[rep].dim

    def trans(self, order):
        s = []
        for i in range(order + 1):
            i = i + 5 if i < 3 else i
            s.append(self.dif_ker[str(i)])
        return torch.cat(s)

    def nlpdo(self, in_type, out_type, order, reduction, s, g, stride=1):
        return nlpdo_torch(self, in_type, out_type, order, reduction, s, g, stride)

    def norm(self, in_type, affine=True, momentum=0.1, track_running_stats=True):
        rep, num_rep = in_type
        dim_rep = self.dim_rep(rep)
        return GroupBatchNorm(num_rep, dim_rep, affine, momentum, track_running_stats)

    def GroupPool(self, in_type, type='max'):
        rep, num_rep = in_type
        return GroupPooling(self.dim_rep(rep), num_rep, type)

    def MaxPool(self, in_type, kernel_size=5):
        rep, num_rep = in_type
        C = self.dim_rep(rep) * num_rep
        return nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=1), Downsample(channels=C, filt_size=kernel_size, stride=2))

    def AvgPool(self, in_type, kernel_size=5):
        rep, num_rep = in_type
        C = self.dim_rep(rep) * num_rep
        return nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=1), Downsample(channels=C, filt_size=kernel_size, stride=2))

