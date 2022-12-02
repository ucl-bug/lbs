from typing import Callable

import flax.linen as nn
import numpy as np
from flax.linen.initializers import normal
from jax import numpy as jnp


class Project(nn.Module):
    in_channels: int = 32
    out_channels: int = 32
    activation: Callable = nn.gelu  # lambda x: jnp.exp(-(x**2)/2)

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.in_channels)(x)
        x = self.activation(x)
        x = nn.Dense(self.in_channels)(x)
        x = self.activation(x)
        x = nn.Dense(self.out_channels)(x)
        return x


def get_grid(x, center=False):
    x1 = jnp.linspace(0, 1, x.shape[1])
    x2 = jnp.linspace(0, 1, x.shape[2])
    if center:
        x1 = x1 - jnp.mean(x1)
        x2 = x2 - jnp.mean(x2)
    x1, x2 = jnp.meshgrid(x1, x2)
    grid = jnp.expand_dims(jnp.stack([x1, x2], axis=-1), 0)
    batched_grid = jnp.repeat(grid, x.shape[0], axis=0)
    return batched_grid


class BNOS(nn.Module):
    width: int = 4
    depth: int = 4
    iterations: int = 1
    channels_last_proj: int = 16
    activation: Callable = nn.gelu
    out_channels: int = 1
    padding: int = 0

    @nn.compact
    def __call__(self, sos, pml, src) -> jnp.ndarray:
        # Concatenate inputs
        k = jnp.concatenate([sos, pml, src], axis=-1)

        # Pad input
        if self.padding > 0:
            k = jnp.pad(
                k,
                ((0, 0), (0, self.padding), (0, self.padding), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        # Generate coordinate grid, and append to input channels
        grid = get_grid(k)
        k = jnp.concatenate([k, grid], axis=-1)

        # Generate the first source term
        u = Project(2 * self.width, self.width)(k)

        # Apply the Born stages
        for _ in range(self.depth):
            u = BornSeries(iterations=self.iterations, activation=self.activation)(u, k)

        # Unpad
        if self.padding > 0:
            u = u[:, : -self.padding, : -self.padding, :]

        # Project to the output channels
        u = Project(self.channels_last_proj, self.out_channels)(u)
        return u


class BornSeries(nn.Module):
    iterations: int = 4
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, u, k) -> jnp.ndarray:
        in_channels = u.shape[-1]
        gamma = Project(2 * in_channels, in_channels**2)(k)
        delta = Project(2 * in_channels, in_channels**2)(k)
        greens = Greens()

        # normalize
        gamma = gamma / gamma.shape[3]
        delta = delta / delta.shape[3]

        # source = u

        for iter in range(self.iterations):
            local_term = 2 * u - apply_matrix(u, gamma)

            delta_u = apply_matrix(u, delta)
            greens_u = apply_matrix(greens(delta_u), gamma)

            u = local_term + greens_u  # + source

            # u = 2u - gamma*u + gamma*G*delta*u

        # Add bias term
        bias = self.param("bias", normal(1.0, jnp.float32), (u.shape[-1],), jnp.float32)

        u = u + bias

        # Use nonlinearity
        u = self.activation(u)
        return u


def apply_matrix(x, matrix):
    size = int(np.round(np.sqrt(matrix.shape[-1])))
    shape = matrix.shape[:-1] + (size, size)
    matrix = jnp.reshape(matrix, shape)
    return jnp.einsum("nhwj,nhwjk->nhwk", x, matrix)


class Greens(nn.Module):
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, u) -> jnp.ndarray:
        # Get freq axis
        kx = jnp.fft.rfftfreq(u.shape[1], d=1.0 / u.shape[1])
        ky = jnp.fft.fftfreq(u.shape[2], d=1.0 / u.shape[2])
        Kx, Ky = jnp.meshgrid(kx, ky)
        K = jnp.stack([Kx, Ky], axis=-1)
        K = jnp.expand_dims(K, 0)
        in_channels = u.shape[-1]
        spectrum_filter_real = Project(2 * in_channels, in_channels**2)(K)
        spectrum_filter_imag = Project(2 * in_channels, in_channels**2)(K)
        spectrum_filter = spectrum_filter_real + 1j * spectrum_filter_imag

        # Normalize to unit norm for gaussians
        spectrum_filter = spectrum_filter / spectrum_filter.shape[3]

        u = jnp.fft.rfftn(u, axes=(1, 2))
        print(u.shape, spectrum_filter.shape)
        Gu = apply_matrix(u, spectrum_filter)
        y = jnp.fft.irfftn(Gu, axes=(1, 2))

        return y
