"""
Reference: https://github.com/zh217/torch-dct
"""

import numpy as np
import torch
import torch.nn as nn

class DCT():
    def __init__(self, resolution, device, norm=None, bias=False):
        self.resolution = resolution
        self.norm = norm
        self.device = device

        I = torch.eye(self.resolution, device=self.device)
        self.forward_transform = nn.Linear(resolution, resolution, bias=bias).to(self.device)
        self.forward_transform.weight.data = self._dct(I, norm=self.norm).data.t()
        self.forward_transform.weight.requires_grad = False

        self.inverse_transform = nn.Linear(resolution, resolution, bias=bias).to(self.device)
        self.inverse_transform.weight.data = self._idct(I, norm=self.norm).data.t()
        self.inverse_transform.weight.requires_grad = False

    def _dct(self, x, norm=None):
        """
        Discrete Cosine Transform, Type II (a.k.a. the DCT)
        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
        :param x: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the DCT-II of the signal over the last dimension
        """
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)

        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

        Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

        k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

        if norm == 'ortho':
            V[:, 0] /= np.sqrt(N) * 2
            V[:, 1:] /= np.sqrt(N / 2) * 2

        V = 2 * V.view(*x_shape)
        return V

    def _idct(self, X, norm=None):
        """
        The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
        Our definition of idct is that idct(dct(x)) == x
        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
        :param X: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the inverse DCT-II of the signal over the last dimension
        """

        x_shape = X.shape
        N = x_shape[-1]

        X_v = X.contiguous().view(-1, x_shape[-1]) / 2

        if norm == 'ortho':
            X_v[:, 0] *= np.sqrt(N) * 2
            X_v[:, 1:] *= np.sqrt(N / 2) * 2

        k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V_t_r = X_v
        V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

        V_r = V_t_r * W_r - V_t_i * W_i
        V_i = V_t_r * W_i + V_t_i * W_r

        V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

        v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
        x = v.new_zeros(v.shape)
        x[:, ::2] += v[:, :N - (N // 2)]
        x[:, 1::2] += v.flip([1])[:, :N // 2]
        return x.view(*x_shape)

    def forward(self, x):
        X1 = self.forward_transform(x)
        X2 = self.forward_transform(X1.transpose(-1, -2))
        return X2.transpose(-1, -2)

    def inverse(self, x):
        X1 = self.inverse_transform(x)
        X2 = self.inverse_transform(X1.transpose(-1, -2))
        return X2.transpose(-1, -2)