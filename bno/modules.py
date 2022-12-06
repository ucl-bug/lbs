import flax.linen as nn
from fno import FNO2D
from jax import numpy as jnp
from jwave import FourierSeries
from jwave.acoustics.time_harmonic import born_series
from jwave.geometry import Domain, Medium

from .models.bno import BNO
from .models.fno import FNO2D


class NeuralOperatorWrapper(nn.Module):
    @nn.compact
    def __call__(
        self, sos: jnp.ndarray, pml: jnp.ndarray, src: jnp.ndarray
    ) -> jnp.ndarray:
        pass


class WrappedFNO(NeuralOperatorWrapper):
    stages: int = 6
    channels: int = 8
    padding: int = 32
    modes: int = 16
    channels_last_proj: int = 128
    output_dtype: jnp.dtype = jnp.complex64

    @nn.compact
    def __call__(
        self, sos: jnp.ndarray, pml: jnp.ndarray, src: jnp.ndarray
    ) -> jnp.ndarray:
        # Two channels if complex
        if self.output_dtype == jnp.complex64:
            out_channels = 2
        else:
            out_channels = 1

        # Concate inputs
        x = jnp.concatenate([sos, pml, src], axis=-1)
        y = FNO2D(
            depth=self.stages,
            width=self.channels,
            out_channels=out_channels,
            padding=self.padding,
            channels_last_proj=2 * self.channels,
            modes1=self.modes,
            modes2=self.modes,
        )(x)

        if self.output_dtype == jnp.complex64:
            return jnp.expand_dims(y[..., 0] + 1j * y[..., 1], -1)
        else:
            return y


class WrappedBNO(nn.Module):
    stages: int = 4
    channels: int = 8
    dtype: jnp.dtype = jnp.complex64
    last_proj: int = 128
    use_nonlinearity: bool = True
    use_grid: bool = True

    @nn.compact
    def __call__(self, sos, pml, src):
        # Two channels if complex
        if self.dtype == jnp.complex64:
            out_channels = 2
        else:
            out_channels = 2

        # Concate inputs
        y = BNO(
            depth=self.stages,
            width=self.channels,
            out_channels=out_channels,
            channels_last_proj=self.last_proj,
            padding=32,
            use_nonlinearity=self.use_nonlinearity,
            use_grid=self.use_grid,
        )(sos, pml, src)

        if True:  # self.dtype == jnp.complex64:
            return jnp.expand_dims(y[..., 0] + 1j * y[..., 1], -1)
        else:
            return jnp.expand_dims(jnp.sqrt(y[..., 0] ** 2 + y[..., 1] ** 2), -1)


class WrappedCBS(nn.Module):
    stages: int = 12

    @nn.compact
    def __call__(self, sos, pml, src):
        # Strip batch dimension and channel dimension
        sos = sos[0, ..., 0]
        src = src[0, ..., 0]

        # Setup domain
        image_size = sos.shape[1]
        N = tuple([image_size] * 2)
        dx = (1.0, 1.0)
        domain = Domain(N, dx)

        # Define fields
        sound_speed = FourierSeries(sos, domain)
        src = FourierSeries(src, domain)

        # Make model
        medium = Medium(domain, sound_speed)

        # Predict
        predicted = born_series(
            medium, src, max_iter=self.stages, k0=0.79056941504209483299972338610818
        ).on_grid

        # Expand dims
        predicted = jnp.expand_dims(predicted, 0)

        return predicted
