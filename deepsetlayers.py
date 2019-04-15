r"""
Permutation Equivariant and Permutation Invariant layers, as described in the
paper Deep Sets, by Zaheer et al. (https://arxiv.org/abs/1703.06114)
"""

import math

import torch
from torch import nn
from torch.nn import init


class EquivLinear(nn.Module):
    r"""Permutation equivariant linear layer.
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``
    """
    def __init__(self, in_features, out_features, bias=True, reduction='average'):
        super(EquivLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        assert reduction in ['mean', 'sum', 'max', 'min'],  \
            '\'reduction\' should be \'mean\'/\'sum\'\'max\'/\'min\', got {}'.format(reduction)
        self.reduction = reduction

        self.alpha = nn.Parameter(torch.Tensor(self.in_features,
                                               self.out_features))
        self.beta = nn.Parameter(torch.Tensor(self.in_features,
                                              self.out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, self.out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.alpha)
        init.xavier_uniform_(self.beta)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.alpha)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, X, sizes=None):
        r"""
        Maps the input set X = {x_1, ..., x_M} to the output set
        Y = {y_1, ..., y_M} through a permutation equivariant linear transformation
        of the form:
            $y_i = \alpha x_i + \beta \sum_j x_j + bias$
        Inputs:
        X: N sets of size at most M where each element has dimension in_features
           (tensor with shape (N, M, in_features))
        sizes: size of each set in X (long tensor with shape (N,) or None); if
            None, all sets have the maximum size M.
            Default: ``None``.
        Outputs:
        Y: N sets of same cardinality as in X where each element has dimension
           out_features (tensor with shape (N, M, out_features))
        """
        N, M, _ = X.shape
        device = X.device
        Y = torch.zeros(N, M, self.out_features).to(device)
        if sizes is None:
            mask = torch.ones(N, M).byte().to(device)
            size_matrix = M*torch.ones(N, 1).float().to(device)
        else:
            mask = (torch.arange(M).reshape(1, -1).to(device) < sizes.reshape(-1, 1))
            size_matrix = sizes.float().unsqueeze(1)

        if self.reduction == 'mean':
            sizes = sizes.float().unsqueeze(1)
            X = X * mask.unsqueeze(2).float()
            Y[mask] = (X @ self.alpha
                       + ((X.sum(dim=1) @ self.beta)/size_matrix).unsqueeze(1)
                      )[mask]

        elif self.reduction == 'sum':
            X = X * mask.unsqueeze(2).float()
            Y[mask] = (X @ self.alpha
                       + (X.sum(dim=1) @ self.beta).unsqueeze(1)
                      )[mask]

        elif self.reduction == 'max':
            Z = X.clone()
            Z[~mask] = float('-Inf')
            Y[mask] = ((X @ self.alpha)
                       + (Z.max(dim=1)[0] @ self.beta).unsqueeze(1)
                      )[mask]

        else:  # min
            Z = X.clone()
            Z[~mask] = float('Inf')
            Y[mask] = (X @ self.alpha
                       + (Z.min(dim=1)[0] @ self.beta).unsqueeze(1)
                      )[mask]

        if self.bias is not None:
            Y[mask] += self.bias

        return Y

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, reduction={}'.format(
            self.in_features, self.out_features,
            self.bias is not None, self.reduction)


class InvLinear(EquivLinear):
    r"""Permutation invariant linear layer.
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``
    """
    def __init__(self, in_features, out_features, bias=True, reduction='average'):
        super(InvLinear, self).__init__(in_features, out_features, bias=bias, reduction=reduction)
        self.reset_parameters()

    def reset_parameters(self):
        super(InvLinear, self).reset_parameters()
        init.zeros_(self.alpha.data)
        self.alpha.requires_grad_(False)

    def forward(self, X, sizes=None):
        r"""
        Maps the input set X = {x_1, ..., x_M} to a vector y of dimension out_features,
        through a permutation invariant linear transformation of the form:
            $y = \beta \sum_j x_j + bias$
        Inputs:
        X: N sets of size at most M where each element has dimension in_features
           (tensor with shape (N, M, in_features))
        sizes: size of each set in X (long tensor with shape (N,) or None); if
            None, all sets have the maximum size M.
            Default: ``None``).
        Outputs:
        Y: N vectors of dimension out_features (tensor with shape (N, out_features))
        """
        return super(InvLinear, self).forward(X, sizes)[:, 0]
