from typing import Callable

import flax.linen as nn
from flax.linen.initializers import normal
from jax import numpy as jnp
from jaxdf.discretization import FourierSeries
from jwave import FourierSeries
from jwave.geometry import Domain

from .utils import CDense, CProject, c_gelu


def homog_greens(field: FourierSeries, k0: object = 1.0, epsilon: object = 0.1):
    freq_grid = field._freq_grid
    p_sq = jnp.sum(freq_grid**2, -1)

    g_fourier = 1.0 / (p_sq - (k0**2) - 1j * epsilon)
    u = field.on_grid[..., 0]
    u_fft = jnp.fft.fftn(u)
    Gu_fft = g_fourier * u_fft
    Gu = jnp.fft.ifftn(Gu_fft)
    return field.replace_params(jnp.expand_dims(Gu, -1))


class CBNO(nn.Module):
    r"""
    Fourier Neural Operator for 2D signals.
    Implemented from
    https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
    Attributes:
      modes1: Number of modes in the first dimension.
      modes2: Number of modes in the second dimension.
      width: Number of channels to which the input is lifted.
      depth: Number of Fourier stages
      channels_last_proj: Number of channels in the hidden layer of the last
        2-layers Fully Connected (channel-wise) network
      activation: Activation function to use
      out_channels: Number of output channels, >1 for non-scalar fields.
    """
    modes1: int = 12
    modes2: int = 12
    width: int = 32
    depth: int = 4
    channels_last_proj: int = 128
    activation: Callable = nn.gelu
    out_channels: int = 1
    padding: int = 32  # Padding for non-periodic inputs

    @nn.compact
    def __call__(self, sos, pml, src) -> jnp.ndarray:
        k = (0.5 / sos) ** 2

        # Pad input
        if self.padding > 0:
            padval = self.param("padval", normal(1.0, jnp.float32), (3,), jnp.float32)

            sos = jnp.pad(
                sos,
                ((0, 0), (0, self.padding), (0, self.padding), (0, 0)),
                mode="constant",
                constant_values=1.0,
            )

            pml = jnp.pad(
                pml,
                ((0, 0), (0, self.padding), (0, self.padding), (0, 0)),
                mode="constant",
                constant_values=1.0,
            )

            src = jnp.pad(
                src,
                ((0, 0), (0, self.padding), (0, self.padding), (0, 0)),
                mode="constant",
                constant_values=0.0,
            )

        # Generate coordinate grid, and append to input channels
        grid = self.get_grid(src)
        k = jnp.concatenate([k, grid, src], axis=-1)
        x = jnp.concatenate([k, grid, src], axis=-1)
        # k = jnp.concatenate([src, sos, pml], axis=-1)
        # x = jnp.concatenate([src, sos, pml], axis=-1)

        # Lift the input to a higher dimension
        x = jnp.zeros_like(src) + 0j

        # Apply Fourier stages, last one has no activation
        # (can't find this in the paper, but is in the original code)
        for depthnum in range(self.depth):
            activation = self.activation if depthnum < self.depth - 1 else lambda x: x
            x = ComplexBornStage(
                out_channels=self.width,
                modes1=self.modes1,
                modes2=self.modes2,
                activation=activation,
            )(x, k)

        # Unpad
        if self.padding > 0:
            x = x[:, : -self.padding, : -self.padding, :]

        # Project to the output channels
        x = CDense(1)(x)
        return x

    @staticmethod
    def get_grid(x):
        x1 = jnp.linspace(0, 1, x.shape[1])
        x2 = jnp.linspace(0, 1, x.shape[2])
        x1, x2 = jnp.meshgrid(x1, x2)
        grid = jnp.expand_dims(jnp.stack([x1, x2], axis=-1), 0)
        batched_grid = jnp.repeat(grid, x.shape[0], axis=0)
        return batched_grid


class ComplexBornStage(nn.Module):
    activation: Callable = c_gelu

    @nn.compact
    def __call__(self, x, k, grid, src):
        k0 = self.param("k0", normal(1.0, jnp.float32), (1,), jnp.float32)[0]
        epsilon = self.param("epsilon", normal(1.0, jnp.float32), (1,), jnp.float32)[0]

        real_in = jnp.concatenate([k, x.real, x.imag, src, grid], -1)
        gamma_1 = CProject(16, 1, self.activation)(real_in)
        gamma_2 = CProject(16, 1, self.activation)(real_in)
        src = CProject(16, 1, self.activation)(real_in)

        # Greens func
        def G(u):
            _params = jnp.zeros(
                list(k0.shape[1:3])
                + [
                    1,
                ]
            )
            field = FourierSeries(_params, Domain(k0.shape[1:3], (1, 1)))
            freq_grid = field._freq_grid
            p_sq = jnp.sum(freq_grid**2, -1)

            g_fourier = jnp.expand_dims(1.0 / (p_sq - (k0**2) - 1j * epsilon), 0)
            u = u[..., 0]
            u_fft = jnp.fft.fftn(u, axis=(1, 2))
            Gu_fft = g_fourier * u_fft
            Gu = jnp.fft.ifftn(Gu_fft, axis=(1, 2))
            Gu = jnp.expand_dims(Gu, -1)
            return Gu

        # Operator
        x = x - gamma_1 * (x - G(gamma_2 * x - src))

        return x


class ComplexSpectralConv2d(nn.Module):
    out_channels: int = 32
    modes1: int = 12
    modes2: int = 12

    @nn.compact
    def __call__(self, x):
        # Getting shapes
        in_channels = x.shape[-1]
        if x.shape[1] % 2 == 0:
            x_size = 2 * self.modes1 + 1
        else:
            x_size = 2 * self.modes1

        if x.shape[2] % 2 == 0:
            y_size = 2 * self.modes2 + 1
        else:
            y_size = 2 * self.modes2

        # Initialize kernel
        kernel_r = self.param(
            "kernel_r",
            normal(stddev=1e-2, dtype=jnp.float32),
            (in_channels, self.out_channels, x_size, y_size),
        )
        kernel_i = self.param(
            "kernel_i",
            normal(stddev=1e-2, dtype=jnp.float32),
            (in_channels, self.out_channels, x_size, y_size),
        )

        kernel = kernel_r + 1j * kernel_i

        # Convolve in spectral domain
        x_ft = jnp.fft.fftn(x, axes=(1, 2))
        out_ft = jnp.zeros_like(x_ft)

        # get the center self.modes modes
        x_centred = jnp.fft.fftshift(x_ft, axes=(1, 2))
        idxc = x_centred.shape[1] // 2
        m = self.modes1
        x_centred = x_centred[:, idxc - m : idxc + m + 1, idxc - m : idxc + m + 1, :]
        inner_part = jnp.einsum("bijc,coij->bijo", x_centred, kernel)
        out_ft = out_ft.at[:, idxc - m : idxc + m + 1, idxc - m : idxc + m + 1, :].set(
            inner_part
        )

        # Restore
        out_ft = jnp.fft.ifftshift(out_ft, axes=(1, 2))
        out = jnp.fft.ifftn(out_ft, axes=(1, 2))
        return out
