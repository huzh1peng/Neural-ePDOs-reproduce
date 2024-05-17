import torch

def kronecker(x, y):
    return torch.einsum('ij,mn->imjn', [x, y]).reshape(x.size(0) * y.size(0), x.size(1) * y.size(1))

def direct_sum(a, b):
    n = a.size(0) + b.size(0)
    out = torch.zeros((n, n), dtype=a.dtype, device=a.device)
    out[:a.size(0), :a.size(0)] = a
    out[a.size(0):, a.size(0):] = b
    return out

def ortho_basis(x):
    x, _ = torch.qr(x.transpose(0, 1))
    return x.transpose(0, 1)

def solve(A):
    n = torch.linalg.matrix_rank(A)
    u, s, v = torch.svd(A, some=True)
    return v[:, n:A.size(1)]
