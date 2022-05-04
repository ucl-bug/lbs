
import flax.linen as nn
from fno import FNO2D
from jax import numpy as jnp

from .models.bno import BNO
from .models.born_unroll import UnrolledBorn
from .models.cbno import CBNO


class WrappedFNO(nn.Module):
  stages: int = 4
  channels: int = 8
  padding: int = 32
  modes: int = 16
  dtype: jnp.dtype = jnp.complex64

  @nn.compact
  def __call__(self, sos, pml, src):
    # Two channels if complex
    if self.dtype == jnp.complex64:
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
      channels_last_proj=2*self.channels,
      modes1=32,
      modes2=32,
    )(x)

    if self.dtype == jnp.complex64:
      return jnp.expand_dims(y[...,0] + 1j*y[...,1], -1)
    else:
      return y

class WrappedUBS(nn.Module):
  stages: int = 12
  project_inner_ch: int = 16
  padding: int = 32
  dtype: jnp.dtype = jnp.complex64

  @nn.compact
  def __call__(self, sos, pml, src, unrolls):
    k_sq = (1/sos)**2

    y = UnrolledBorn(
      stages=self.stages,
      project_inner_ch=self.project_inner_ch,
      padding=self.padding,
    )(k_sq, src, unrolls)

    if self.dtype == jnp.complex64:
      return y
    else:
      return jnp.abs(y)


class WrappedComplexBNO(nn.Module):
  stages: int = 4
  channels: int = 8
  dtype: jnp.dtype = jnp.complex64

  @nn.compact
  def __call__(self, sos, pml, src):
    # Two channels if complex
    if self.dtype == jnp.complex64:
      out_channels = 2
    else:
      out_channels = 2

    # Concate inputs
    y = CBNO(
      depth=self.stages,
      width=self.channels,
      out_channels=out_channels,
      padding=32,
      modes1=32,
      modes2=32,
    )(sos, pml, src)

    if self.dtype == jnp.complex64:
      return y
    else:
      return jnp.abs(y)


class WrappedBNO(nn.Module):
  stages: int = 4
  channels: int = 8
  padding: int = 32
  dtype: jnp.dtype = jnp.complex64

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
      padding=self.padding,
      modes1=32,
      modes2=32,
    )(sos, pml, src)

    if self.dtype == jnp.complex64:
      return jnp.expand_dims(y[...,0] + 1j*y[...,1], -1)
    else:
      return jnp.expand_dims(jnp.sqrt(y[...,0]**2 + y[...,1]**2), -1)
