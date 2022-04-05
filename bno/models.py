from typing import Callable

import flax.linen as nn
from flax.linen.initializers import normal
from jax import numpy as jnp


class LiftTransformation(nn.Module):
    channels: int = 32

    @nn.compact
    def __call__(self, x):
        # The transformation works in the channels dimension
        x = nn.Dense(self.channels)(x)
        return x

class ProjectBack(nn.Module):
    out_channels: int = 2
    internal_channels: int = 128
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.internal_channels)(x)
        x = self.activation(x)
        x = nn.Dense(self.out_channels)(x)
        return x

class FourierStage(nn.Module):
    channels: int = 32
    kmax: int = 16
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x):
        # x      -> [batch, height, width, in_channels]
        # kernel -> [in_channels, out_channels, kmax, kmax]
        kernel = self.param(
            'kernel',
            normal(1/self.channels, jnp.complex64),
            (x.shape[-1], self.channels, self.kmax, self.kmax),
            jnp.complex64
        )

        # Compute FFT
        x_ft = jnp.fft.rfftn(x, axes=(1, 2))

        # Multiply by kernel
        padsize_1 = x_ft.shape[1] - self.kmax
        padsize_2 = x_ft.shape[2] - self.kmax
        kernel_padded = jnp.pad(
            kernel,
            ((0,0), (0,0), (0, padsize_1), (0, padsize_2)),
            "constant",
            constant_values = 0
        )
        # output -> [batch, height, width, channels]
        x_filt = jnp.einsum('bijc,coij->bijo', x_ft, kernel_padded)

        # Inverse FFT
        x_filt = jnp.fft.irfftn(x_filt, axes=(1, 2))

        # Add a convolution
        x_conv = nn.Conv(self.channels,(1,1))(x)

        # Put together with non linearity
        y = self.activation(x_filt + x_conv)

        return y

class FNO(nn.Module):
    stages: int = 6
    channels: int = 32
    kmax: int = 16

    @nn.compact
    def __call__(self, x):
        # Lift to higher dimension
        x = LiftTransformation(self.channels)(x)

        # Apply fourier stages
        for stagenum in range(self.stages):
            x = FourierStage(self.channels, self.kmax)(x)

        # Transform back to lower dimension
        x = ProjectBack(2, 128)(x)

        # Make complex
        x = jnp.expand_dims(x[...,0] + 1j*x[...,1], -1)
        return x
