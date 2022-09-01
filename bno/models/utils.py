from typing import Callable

from flax import linen as nn
from jax import lax
from jax import numpy as jnp
from jax import random


def normal(stddev=1e-2, dtype=jnp.float32) -> Callable:
    def init(key, shape, dtype=dtype):
        keys = random.split(key)
        return random.normal(keys[0], shape) * stddev

    return init


def constant(value=1e-2, dtype=jnp.float32) -> Callable:
    def init(key, shape, dtype=dtype):
        keys = random.split(key)
        return random.normal(keys[0], shape) * 0.0 + value

    return init


def pad_constant(
    x: jnp.ndarray,
    pad_width: int,
    constant_value: int = 0,
    mode: str = "side",
):
    r"""Pad an array with a constant value. It works
    with 4D arrays with dimension NWHC. Only pads the
    dimensions W and H.

    Arguments:
      x (jnp.ndarray): The array to pad.
      pad_width (int): The number of pixels to pad.
      constant_value (int): The value to pad with.

    Returns:
      jnp.ndarray: The padded array.
    """
    assert mode in ["side", "symmetric"], "Only side and symmetric modes are supported."

    if mode == "side":
        return jnp.pad(
            x,
            ((0, 0), (0, pad_width), (0, pad_width), (0, 0)),
            mode="constant",
            constant_values=constant_value,
        )
    elif mode == "symmetric":
        return jnp.pad(
            x,
            ((0, 0), (pad_width, pad_width), (pad_width, pad_width), (0, 0)),
            mode="constant",
            constant_values=constant_value,
        )


def unpad(
    x: jnp.ndarray,
    pad_width: int,
    mode: str = "side",
):
    r"""Unpad an array. It works with 4D arrays with dimension NWHC.
    Only unpads the dimensions W and H.

    Arguments:
      x (jnp.ndarray): The array to unpad.
      pad_width (int): The number of pixels to unpad.

    Returns:
      jnp.ndarray: The unpadded array.
    """
    assert mode in ["side", "symmetric"], "Only side and symmetric modes are supported."

    if mode == "side":
        return x[:, :-pad_width, :-pad_width, :]
    elif mode == "symmetric":
        return x[:, pad_width:-pad_width, pad_width:-pad_width, :]


class CProject(nn.Module):
    in_channels: int = 32
    out_channels: int = 32
    activation: Callable = lambda x: jnp.exp(-(x**2))

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.in_channels)(x)
        x = self.activation(x)
        x = nn.Dense(self.in_channels)(x)
        x = self.activation(x)
        x = CDense(self.out_channels)(x)
        return x


def c_gelu(x):
    return nn.gelu(x.real) + 1j * nn.gelu(x.imag)


class CDense(nn.Module):
    features: int
    use_bias: bool = True
    dtype: jnp.dtype = jnp.complex64

    @nn.compact
    def __call__(self, inputs):
        inputs = jnp.asarray(inputs, self.dtype)
        kernel_r = self.param(
            "kernel_r",
            normal(stddev=1e-2, dtype=jnp.float32),
            (inputs.shape[-1], self.features),
        )
        kernel_i = self.param(
            "kernel_i",
            normal(stddev=1e-2, dtype=jnp.float32),
            (inputs.shape[-1], self.features),
        )
        kernel = kernel_r + 1j * kernel_i
        y = lax.dot_general(inputs, kernel, (((inputs.ndim - 1,), (0,)), ((), ())))
        if self.use_bias:
            bias_r = self.param(
                "bias_r", normal(stddev=1e-2, dtype=jnp.float32), (self.features,)
            )
            bias_i = self.param(
                "bias_i", normal(stddev=1e-2, dtype=jnp.float32), (self.features,)
            )
            bias = bias_r + 1j * bias_i
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y


def get_grid(x):
    r"""Given an input array of dimension [B, H, W, C], it
    returns a grid of dimension [B, H, W, 2] with the
    coordinates of each pixel in the image.

    Arguments:
      x (jnp.ndarray): The input array.

    Returns:
      jnp.ndarray: The grid.
    """
    x1 = jnp.linspace(0, 1, x.shape[1])
    x2 = jnp.linspace(0, 1, x.shape[2])
    x1, x2 = jnp.meshgrid(x1, x2)
    grid = jnp.expand_dims(jnp.stack([x1, x2], axis=-1), 0)
    batched_grid = jnp.repeat(grid, x.shape[0], axis=0)
    return batched_grid
