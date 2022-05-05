import flax.linen as nn
from jax import numpy as jnp
from jaxdf import FourierSeries
from jaxdf.geometry import Domain

from .utils import CProject, constant, pad_constant, unpad


class UnrolledBorn(nn.Module):
  stages: int = 12
  project_inner_ch: int = 32
  padding: int = 32

  @nn.compact
  def __call__(self, k_sq, src, unrolls):
    # Pad fields to accomodate pml
    k_sq = pad_constant(k_sq, self.padding, jnp.min(k_sq), 'symmetric')
    src = pad_constant(src, self.padding, 0.0, 'symmetric')

    # Get grid
    grid = self.get_grid(src)

    # Group inputs
    all_ins = jnp.concatenate([k_sq, grid, src.real, src.imag], axis=-1)
    # Initialize learnable initial guess field
    u_r = self.param('u0_r', constant(0., jnp.float32), src.shape[1:], jnp.float32)
    u_i = self.param('u0_i', constant(0., jnp.float32), src.shape[1:], jnp.float32)
    u0 = u_r + 1j*u_i

    u0 = jnp.expand_dims(u0, 0) * 0.0 # TODO: REMOVE THIS
    u0 = jnp.repeat(u0, src.shape[0], axis=0) # Batching initial field

    # Apply Born-like stages
    unrolls = min([unrolls, self.stages])
    for i in range(unrolls):
      u0 = BornStage(self.project_inner_ch)(all_ins, u0)

    # Remove padding
    u0 = unpad(u0, self.padding, 'symmetric')

    return u0

  @staticmethod
  def get_grid(x):
    x1 = jnp.linspace(0, 1, x.shape[1])
    x2 = jnp.linspace(0, 1, x.shape[2])
    x1, x2 = jnp.meshgrid(x1, x2)
    grid = jnp.expand_dims(jnp.stack([x1, x2], axis=-1), 0)
    batched_grid = jnp.repeat(grid, x.shape[0], axis=0)
    return batched_grid

class BornStage(nn.Module):
  project_inner_ch: int = 32

  @nn.compact
  def __call__(self, all_ins, uk):
    # Concat real field to inputs
    full_in = jnp.concatenate([all_ins, uk.real, uk.imag], axis=-1)

    # Get gammas
    gamma_1 = CProject(self.project_inner_ch, 1)(full_in)
    gamma_2 = CProject(self.project_inner_ch, 1)(full_in)
    src = CProject(self.project_inner_ch, 1)(full_in)
    u_a = CProject(self.project_inner_ch, 1)(full_in)
    u_b = CProject(self.project_inner_ch, 1)(full_in)
    G = TunableGreens()

    # Apply Born-like stage
    u1 = gamma_2*u_a + src
    u2 = gamma_1*(u_b - G(u1))
    u_new = uk - u2
    return u_new

class TunableGreens(nn.Module):

  @nn.compact
  def __call__(self, u):
      # Stage params
      k0 = self.param('k0', constant(1., jnp.float32), (1,), jnp.float32)[0]
      epsilon = self.param('epsilon', constant(1., jnp.float32), (1,), jnp.float32)[0]
      freq_scale = self.param('freq_scale', constant(1., jnp.float32), (1,), jnp.float32)[0]

      # Keep them positive
      k0 = nn.softplus(k0)
      epsilon = nn.softplus(epsilon)
      freq_scale = nn.softplus(freq_scale)

      # Get frequency axis squared
      _params = jnp.zeros(list(u.shape[1:3]) + [1,]) + 0j
      field = FourierSeries(_params, Domain(u.shape[1:3], (1,1)))
      freq_grid = field._freq_grid
      p_sq = jnp.sum(freq_grid**2, -1)*freq_scale

      # Apply green's function
      g_fourier = jnp.expand_dims(1.0 / (p_sq - k0 - 1j*(1e-4 + epsilon)), 0)
      u = u[...,0]
      u_fft = jnp.fft.fftn(u, axes=(1,2))
      Gu_fft = g_fourier * u_fft
      Gu = jnp.fft.ifftn(Gu_fft, axes=(1,2))
      Gu = jnp.expand_dims(Gu, -1)

      return Gu
