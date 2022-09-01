import flax.linen as nn
from fno import FNO2D
from jax import numpy as jnp

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
