import flax.linen as nn
from jax import numpy as jnp
from jaxdf import FourierSeries
from jaxdf.geometry import Domain

from .utils import CProject, constant, pad_constant, unpad


class UnrolledBornBase(nn.Module):
    stages: int = 12
    project_inner_ch: int = 32
    padding: int = 32
    size: int = 128

    def setup(self):
        src = jnp.zeros([1, self.size, self.size, 1]) + 0j
        k_sq = src.real

        # Pad fields to accomodate pml
        k_sq = pad_constant(k_sq, self.padding, jnp.max(k_sq), "symmetric")
        src = pad_constant(src, self.padding, 0.0, "symmetric")

        # Get grid
        grid = self.get_grid(src)

        # Apply Born-like stages
        unrolls = self.stages
        self.init_stage = InitialStage(self.project_inner_ch)

        u_new = self.init_stage(src, k_sq)
        u = u_new
        born_stages = list()
        for i in range(unrolls):
            born_stages.append(BornStage(self.project_inner_ch))
        self.born_stages = born_stages

    def prepare_input(self, k_sq, src):
        # Pad fields to accomodate pml
        k_sq = pad_constant(k_sq, self.padding, jnp.max(k_sq), "symmetric")
        src = pad_constant(src, self.padding, 0.0, "symmetric")

        # Get grid
        grid = self.get_grid(src)

        # Group inputs
        all_ins = jnp.concatenate([k_sq, grid, src.real, src.imag], axis=-1)
        return all_ins

    def __call__(self, k_sq, src, unrolls):
        # Pad fields to accomodate pml
        k_sq = pad_constant(k_sq, self.padding, jnp.max(k_sq), "symmetric")
        src = pad_constant(src, self.padding, 0.0, "symmetric")

        # Get grid
        grid = self.get_grid(src)

        # Apply Born-like stages
        unrolls = min([unrolls, self.stages])
        u_new = self.init_stage(src, k_sq)
        u = u_new
        for i in range(unrolls):
            u_new = self.born_stages[i](k_sq, u_new, src, grid)
            u = u + u_new

        # Remove padding
        u = unpad(u, self.padding, "symmetric")

        return u

    def run_with_intermediate(self, k_sq, src, unrolls):
        _intermediate = {
            "fields": [],
            "updates": [],
            "operators": [],
        }

        # Pad fields to accomodate pml
        k_sq = pad_constant(k_sq, self.padding, jnp.max(k_sq), "symmetric")
        src = pad_constant(src, self.padding, 0.0, "symmetric")

        # Get grid
        grid = self.get_grid(src)

        # Apply Born-like stages
        unrolls = min([unrolls, self.stages])
        u_new = self.init_stage(src, k_sq)
        u = u_new
        for i in range(unrolls):
            u_new, gammas = self.born_stages[i].run_with_intermediate(
                k_sq, u_new, src, grid
            )
            u = u + u_new
            _intermediate["fields"].append(u)
            _intermediate["updates"].append(u_new)
            _intermediate["operators"].append(gammas)

        # Remove padding
        u = unpad(u, self.padding, "symmetric")

        return u, _intermediate

    @staticmethod
    def get_grid(x):
        x1 = jnp.linspace(0, 64, x.shape[1]) - 32
        x2 = jnp.linspace(0, 64, x.shape[2]) - 32
        x1, x2 = jnp.meshgrid(x1, x2)
        grid = jnp.expand_dims(jnp.stack([x1, x2], axis=-1), 0)
        batched_grid = jnp.repeat(grid, x.shape[0], axis=0)
        return batched_grid


class InitialStage(nn.Module):
    project_inner_ch: int = 32

    @nn.compact
    def __call__(self, src, k):
        # Get gammas
        gamma_1 = CProject(self.project_inner_ch, 1)(k)
        gamma_2 = CProject(self.project_inner_ch, 1)(k)
        G = TunableGreens()

        return gamma_2 * G(gamma_1 * src)


class BornStage(nn.Module):
    project_inner_ch: int = 32

    def setup(self):
        self.g1 = CProject(self.project_inner_ch, 1)
        self.g2 = CProject(self.project_inner_ch, 1)
        self.g3 = CProject(self.project_inner_ch, 1)
        self.G = TunableGreens()

        self.k1 = CProject(self.project_inner_ch, 1)

    def __call__(self, k_sq, u_new, src, grid):
        # Concat real field to inputs
        full_in = jnp.concatenate([k_sq, src.real, src.imag, grid], axis=-1)

        # Get gammas
        gamma_1 = self.g1(full_in)
        gamma_2 = self.g2(full_in)
        gamma_3 = self.g3(full_in)

        u_new = u_new + self.k1(full_in) * src

        # Apply Born-like stage
        return gamma_2 * self.G(gamma_1 * u_new) + gamma_3 * u_new

    def run_with_intermediate(self, k_sq, u_new, src, grid):
        _intermediate = dict()

        # Concat real field to inputs
        full_in = jnp.concatenate([k_sq, src.real, src.imag, grid], axis=-1)

        # Get gammas
        gamma_1 = self.g1(full_in)
        gamma_2 = self.g2(full_in)
        gamma_3 = self.g3(full_in)

        # Apply Born-like stage
        u_new = u_new + self.k1(full_in) * src
        u_new = gamma_2 * self.G(gamma_1 * u_new) + gamma_3 * u_new

        _intermediate["M1"] = gamma_1
        _intermediate["M2"] = gamma_2
        _intermediate["src"] = gamma_3

        return u_new, _intermediate


class TunableGreens(nn.Module):
    def setup(self):
        self.k0 = self.param("k0", constant(1.0, jnp.float32), (1,), jnp.float32)[0]
        self.epsilon = self.param(
            "epsilon", constant(1.0, jnp.float32), (1,), jnp.float32
        )[0]
        self.amplitude = self.param(
            "amplitude", constant(1.0, jnp.float32), (1,), jnp.float32
        )[0]

    def __call__(self, u):
        # Keep them positive
        k0 = nn.softplus(self.k0)
        epsilon = nn.softplus(self.epsilon)
        amplitude = nn.softplus(self.amplitude)

        # Get frequency axis squared
        _params = (
            jnp.zeros(
                list(u.shape[1:3])
                + [
                    1,
                ]
            )
            + 0j
        )
        field = FourierSeries(_params, Domain(u.shape[1:3], (1, 1)))
        freq_grid = field._freq_grid
        p_sq = amplitude * jnp.sum(freq_grid**2, -1)

        # Apply green's function
        g_fourier = jnp.expand_dims(1.0 / (p_sq - k0 - 1j), 0)
        u = u[..., 0]
        u_fft = jnp.fft.fftn(u, axes=(1, 2))
        Gu_fft = g_fourier * u_fft
        Gu = jnp.fft.ifftn(Gu_fft, axes=(1, 2))
        Gu = jnp.expand_dims(Gu, -1)

        return Gu
