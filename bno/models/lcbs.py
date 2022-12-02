from typing import Callable

import flax.linen as nn
from jax import numpy as jnp
from jax import random
from jaxdf import FourierSeries
from jaxdf.geometry import Domain

from .utils import constant

# Makes all parameters of equation (13) learned. Spatial maps are dependent on the
# context vector
# The parameters of the Born series are decouples, such that fine-tuning is possible


def pad_fun(x, padding, mode="constant"):
    return jnp.pad(x, ((0, 0), (0, padding), (0, padding), (0, 0)), mode=mode)


def pad_no_batch_fun(x, padding, mode="constant"):
    return jnp.pad(x, ((0, padding), (0, padding), (0, 0)), mode=mode)


def unpad_no_batch_fun(x, padding):
    return x[:-padding, :-padding, :]


def unpad_fun(x, padding):
    return x[:, :-padding, :-padding, :]


def init_matrix_field(rng, field):
    # Keep only HxWxC dimension
    shape = field.shape[-3:]

    # Find standard deviation
    stdev = 1 / (shape[-3] ** 2)

    # Make a random matrix field of the same size
    shape = shape + (shape[-1],)
    params = random.normal(rng, shape) * stdev
    return params


class LCBS(nn.Module):
    width: int = 4
    depth: int = 8
    padding: int = 0
    out_channels: int = 2
    channels_last_proj: int = 16
    activation: Callable = nn.gelu  # lambda x: jnp.exp(-(x**2)/2)

    @nn.compact
    def __call__(self, sos, pml, src, gammas=None) -> jnp.ndarray:
        # Pad input
        if self.padding > 0:
            src = pad_fun(src, self.padding, "constant")
            sos = pad_fun(sos, self.padding, "edge")
            pml = pad_fun(pml, self.padding, "edge")

        # Concat with grid
        grid = self.get_grid(src)
        context = jnp.concatenate([sos, grid, pml, src], axis=-1)

        # Normalize
        # context = nn.LayerNorm()(context)

        # Prepare Born iteration inputs (current_estimate, next_source_field)
        # Those tensor are defined on a higher dimension (n-D vector fields)
        x = nn.Dense(self.width)(context)
        x_new = nn.Dense(self.width)(context)

        if gammas is None:
            gammas = FieldsBuilder(depth=self.depth, width=self.width)(context)

        # Apply the born iteration.
        for depthnum in range(self.depth):
            this_gamma = gammas[depthnum]
            x_new = nn.remat(BornIteration)()(x_new, context, this_gamma)
            x = x_new + x

        # Remove padding
        x = unpad_fun(x, self.padding)

        # Project back
        x = nn.Dense(self.channels_last_proj)(x)
        x = self.activation(x)
        x = nn.Dense(self.out_channels)(x)

        return x

    @staticmethod
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


def D(W, x):
    q = jnp.einsum("bijc,bijco->bijo", x, W)
    return q


class BornIteration(nn.Module):
    # Implements the equation
    # x_new = (g1*G*g2 + g3)*x + g4*Gs*s
    @nn.compact
    def __call__(self, u, k, gammas):
        num_channels = u.shape[-1]
        # Source term
        s = Project(num_channels, num_channels)(k)

        g1 = gammas["g1"]
        g2 = gammas["g2"]
        g3 = gammas["g3"]
        g4 = gammas["g4"]

        # Reshape gammas
        reshape_fun = lambda x: jnp.reshape(
            x, [-1, u.shape[1], u.shape[2], u.shape[-1], u.shape[-1]]
        )
        g1 = reshape_fun(g1)
        g2 = reshape_fun(g2)
        g3 = reshape_fun(g3)
        g4 = reshape_fun(g4)

        # Green functions
        G = TunableGreens(num_channels)
        Gs = TunableGreens(num_channels)

        # Apply
        u1 = D(g4, Gs(s))
        u2 = D(g1, G(D(g2, u))) + D(g3, u)
        u_new = u1 + u2

        # Here's where I can put an activation function
        # out = nn.Dense(out.shape[-1])(u_new)
        # out = self.activation(out)
        # out = nn.Dense(out.shape[-1])(out)
        return u_new


@nn.remat
class TunableGreens(nn.Module):
    out_channels: int = 32

    @nn.compact
    def __call__(self, x):
        # Initialize params
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
        u_fft = jnp.fft.fftn(x, axes=(0, 1))
        u_filtered = jnp.einsum("bijc,coij->bijo", u_fft, g_fourier)

        # Back to real values
        Gu = jnp.fft.ifftn(u_filtered, axes=(1, 2)).real

        return Gu


class FieldsBuilder(nn.Module):
    width: int = 4
    depth: int = 8

    @nn.compact
    def __call__(self, context):
        # Generate fields for each iteration of the born series
        params = []
        for depthnum in range(self.depth):
            g1 = Project(2 * self.width, self.width**2)(context)
            g2 = Project(2 * self.width, self.width**2)(context)
            g3 = Project(2 * self.width, self.width**2)(context)
            g4 = Project(2 * self.width, self.width**2)(context)

            params = params + [
                {"g1": g1, "g2": g2, "g3": g3, "g4": g4},
            ]

        return params
