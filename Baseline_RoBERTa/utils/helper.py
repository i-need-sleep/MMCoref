import torch

def binary_to_onehot(x, device):
    # binary (n,1) to onehot (n,2)
    # When I say onehot, I mean the corresponding pred format for a onehot goundtruth
    x = x.repeat(1,2)
    x[:,1] *= torch.tensor([-1]).to(device)
    x[:,1] += torch.tensor([1]).to(device)
    return x