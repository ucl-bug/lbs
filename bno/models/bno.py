from typing import Callable

import flax.linen as nn
from jax import numpy as jnp
from jaxdf import FourierSeries
from jaxdf.geometry import Domain

from .utils import constant


class TunableGreens(nn.Module):
    out_channels: int = 32

    @nn.compact
    def __call__(self, x):
        # Initialize parameters
        in_channels = x.shape[-1]

        k0 = self.param(
            "k0",
            constant(1.0, jnp.float32),
            (in_channels, self.out_channels, 1, 1),
            jnp.float32,
        )
        amplitude = self.param(
            "amplitude",
            constant(1.0, jnp.float32),
            (in_channels, self.out_channels, 1, 1),
            jnp.float32,
        )

        # Keep them positive
        k0 = nn.softplus(k0)
        amplitude = nn.softplus(amplitude)

        # Get frequency axis squared
        _params = (
            jnp.zeros(
                list(x.shape[1:3])
                + [
                    1,
                ]
            )
            + 0j
        )
        field = FourierSeries(_params, Domain(x.shape[1:3], (1, 1)))
        freq_grid = field._freq_grid
        p_sq = amplitude * jnp.sum(freq_grid**2, -1)

        # Apply mixing Green's function
        g_fourier = 1.0 / (p_sq - k0 - 1j)  # [ch, ch, h, w]
        u_fft = jnp.fft.fftn(x, axes=(1, 2))
        u_filtered = jnp.einsum("bijc,coij->bijo", u_fft, g_fourier)

        Gu = jnp.fft.ifftn(u_filtered, axes=(1, 2)).real

        return Gu


class Project(nn.Module):
    in_channels: int = 32
    out_channels: int = 32
    activation: Callable = lambda x: jnp.exp(-(x**2))  # nn.gelu

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.in_channels)(x)
        x = self.activation(x)
        x = nn.Dense(self.in_channels)(x)
        x = self.activation(x)
        x = nn.Dense(self.out_channels)(x)
        return x


def D(x, W):
    q = jnp.einsum("bijc,bijco->bijo", x, W)
    return q


class FourierStage(nn.Module):
    out_channels: int = 32
    activation: Callable = nn.gelu
    use_nonlinearity: bool = True

    @nn.compact
    def __call__(self, x, context):
        gamma1 = Project(self.out_channels, self.out_channels**2)(context)
        gamma2 = Project(self.out_channels, self.out_channels**2)(context)
        gamma3 = Project(self.out_channels, self.out_channels**2)(context)
        G = TunableGreens(self.out_channels)

        # G = SpectralConv2d(
        #  out_channels=self.out_channels,
        #  modes1=self.modes1,
        #  modes2=self.modes2
        # )

        gamma1 = jnp.reshape(
            gamma1,
            (
                gamma1.shape[0],
                gamma1.shape[1],
                gamma1.shape[2],
                self.out_channels,
                self.out_channels,
            ),
        )
        gamma2 = jnp.reshape(
            gamma2,
            (
                gamma2.shape[0],
                gamma2.shape[1],
                gamma2.shape[2],
                self.out_channels,
                self.out_channels,
            ),
        )
        gamma3 = jnp.reshape(
            gamma3,
            (
                gamma3.shape[0],
                gamma3.shape[1],
                gamma3.shape[2],
                self.out_channels,
                self.out_channels,
            ),
        )

        x_fourier = D(G(D(x, gamma2)), gamma1)
        x_local = D(x, gamma3)
        out = x_fourier + x_local

        # Apply nonlinearity
        if self.use_nonlinearity:
            out = nn.Dense(out.shape[-1])(out)
            out = self.activation(out)

        out = nn.Dense(out.shape[-1])(out)
        return out


class BNO(nn.Module):
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
    width: int = 32
    depth: int = 4
    channels_last_proj: int = 128
    activation: Callable = nn.gelu
    out_channels: int = 1
    padding: int = 0  # Padding for non-periodic inputs
    use_nonlinearity: bool = True
    use_grid: bool = True

    @nn.compact
    def __call__(self, sos, pml, src) -> jnp.ndarray:
        # Pad input
        if self.padding > 0:
            src = jnp.pad(
                src,
                ((0, 0), (0, self.padding), (0, self.padding), (0, 0)),
                mode="constant",
            )
            sos = jnp.pad(
                sos, ((0, 0), (0, self.padding), (0, self.padding), (0, 0)), mode="edge"
            )

        # Generate coordinate grid, and append to input channels
        if self.use_grid:
            grid = self.get_grid(src)
            context = jnp.concatenate([sos, grid], axis=-1)
        else:
            context = sos

        # Lift the input to a higher dimension
        x = nn.Dense(self.width)(src) * 0.0
        x_new = nn.Dense(self.width)(src)

        # Apply Fourier stages, last one has no activation
        # (can't find this in the paper, but is in the original code)
        for depthnum in range(self.depth):
            activation = self.activation if depthnum < self.depth - 1 else lambda x: x
            x_new = nn.remat(FourierStage)(
                out_channels=self.width,
                activation=activation,
                use_nonlinearity=self.use_nonlinearity,
            )(x_new, context)
            x = x_new + x

        # Unpad
        if self.padding > 0:
            x = x[:, : -self.padding, : -self.padding, :]

        # Project to the output channels
        x = nn.Dense(self.channels_last_proj)(x)
        x = self.activation(x)
        x = nn.Dense(self.out_channels)(x)

        return x

    @staticmethod
    def get_grid(x):
        x1 = jnp.linspace(0, 1, x.shape[1])
        x2 = jnp.linspace(0, 1, x.shape[2])
        x1, x2 = jnp.meshgrid(x1, x2)
        grid = jnp.expand_dims(jnp.stack([x1, x2], axis=-1), 0)
        batched_grid = jnp.repeat(grid, x.shape[0], axis=0)
        return batched_grid
