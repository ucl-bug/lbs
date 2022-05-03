from typing import Callable

import flax.linen as nn
from flax.linen.initializers import normal
from fno import FNO2D
from jax import lax
from jax import numpy as jnp
from jax import random
from jwave import FourierSeries
from jwave.geometry import Domain

from .cbs import born_solver


class WrappedFNO(nn.Module):
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
      out_channels = 1

    # Concate inputs
    x = jnp.concatenate([sos, pml, src], axis=-1)
    y = FNO2D(
      depth=self.stages,
      width=self.channels,
      out_channels=out_channels,
      padding=self.padding,
    )(x)

    if self.dtype == jnp.complex64:
      return jnp.expand_dims(y[...,0] + 1j*y[...,1], -1)
    else:
      return y

class WrappedCBS(nn.Module):
  stages: int = 32
  padding: int = 64
  alpha: float = 1.0

  @nn.compact
  def __call__(self, sos, pml, src):
    domain = Domain(sos.shape[:-1], (1,1))

    def _solver(sos, src):
      src = FourierSeries(src, domain)
      sos = FourierSeries(sos, domain)
      return born_solver(
        sos,
        src,
        omega=1.0,
        k0=1.0,
        pml_size=self.padding,
        max_iter=self.stages,
        tol=0.0,
        alpha=self.alpha
      )

    solutions, _ = _solver(sos, src)

    return solutions

def normal(stddev=1e-2, dtype = jnp.float32) -> Callable:
  def init(key, shape, dtype=dtype):
    keys = random.split(key)
    return random.normal(keys[0], shape) * stddev
  return init

class ComplexSpectralConv2d(nn.Module):
  out_channels: int = 32
  modes1: int = 12
  modes2: int = 12

  @nn.compact
  def __call__(self, x):
    # Getting shapes
    in_channels = x.shape[-1]
    if x.shape[1] % 2 == 0:
      x_size = 2*self.modes1 + 1
    else:
      x_size = 2*self.modes1

    if x.shape[2] % 2 == 0:
      y_size = 2*self.modes2 + 1
    else:
      y_size = 2*self.modes2

    # Initialize kernel
    kernel_r = self.param(
      "kernel_r",
      normal(stddev=1e-2, dtype=jnp.float32),
      (in_channels, self.out_channels, x_size, y_size)
    )
    kernel_i = self.param(
      "kernel_i",
      normal(stddev=1e-2, dtype=jnp.float32),
      (in_channels, self.out_channels, x_size, y_size)
    )

    kernel = kernel_r + 1j*kernel_i

    # Convolve in spectral domain
    x_ft = jnp.fft.fftn(x, axes=(1, 2))
    out_ft = jnp.zeros_like(x_ft)

    # get the center self.modes modes
    x_centred = jnp.fft.fftshift(x_ft, axes=(1, 2))
    idxc = x_centred.shape[1]//2
    m = self.modes1
    x_centred = x_centred[:, idxc-m:idxc+m+1, idxc-m:idxc+m+1, :]
    inner_part = jnp.einsum(
      'bijc,coij->bijo',
      x_centred,
      kernel
    )
    out_ft = out_ft.at[:, idxc-m:idxc+m+1, idxc-m:idxc+m+1, :].set(inner_part)

    # Restore
    out_ft = jnp.fft.ifftshift(out_ft, axes=(1, 2))
    out = jnp.fft.ifftn(out_ft, axes=(1, 2))
    return out

class SpectralConv2d(nn.Module):
  out_channels: int = 32
  modes1: int = 12
  modes2: int = 12

  @nn.compact
  def __call__(self, x):
    # x.shape: [batch, height, width, in_channels]

    # Initialize parameters
    in_channels = x.shape[-1]
    scale = 1/(in_channels * self.out_channels)
    in_channels = x.shape[-1]
    height = x.shape[1]
    width = x.shape[2]

    # Checking that the modes are not more than the input size
    assert self.modes1 <= height//2 + 1
    assert self.modes2 <= width//2 + 1
    assert height % 2 == 0 # Only tested for even-sized inputs
    assert width % 2 == 0 # Only tested for even-sized inputs

    # The model assumes real inputs and therefore uses a real
    # fft. For a 2D signal, the conjugate symmetry of the
    # transform is exploited to reduce the number of operations.
    # Given an input signal of dimesions (N, C, H, W), the
    # output signal will have dimensions (N, C, H, W//2+1).
    # Therefore the kernel weigths will have different dimensions
    # for the two axis.
    kernel_1_r = self.param(
      'kernel_1_r',
      normal(scale, jnp.float32),
      (in_channels, self.out_channels, self.modes1, self.modes2),
      jnp.float32
    )
    kernel_1_i = self.param(
      'kernel_1_i',
      normal(scale, jnp.float32),
      (in_channels, self.out_channels, self.modes1, self.modes2),
      jnp.float32
    )
    kernel_2_r = self.param(
      'kernel_2_r',
      normal(scale, jnp.float32),
      (in_channels, self.out_channels, self.modes1, self.modes2),
      jnp.float32
    )
    kernel_2_i = self.param(
      'kernel_2_i',
      normal(scale, jnp.float32),
      (in_channels, self.out_channels, self.modes1, self.modes2),
      jnp.float32
    )

    # Perform fft of the input
    x_ft = jnp.fft.rfftn(x, axes=(1, 2))

    # Multiply the center of the spectrum by the kernel
    out_ft = jnp.zeros_like(x_ft)
    s1 = jnp.einsum(
      'bijc,coij->bijo',
      x_ft[:, :self.modes1, :self.modes2, :],
      kernel_1_r + 1j*kernel_1_i)
    s2 = jnp.einsum(
      'bijc,coij->bijo',
      x_ft[:, -self.modes1:, :self.modes2, :],
      kernel_2_r + 1j*kernel_2_i)
    out_ft = out_ft.at[:, :self.modes1, :self.modes2, :].set(s1)
    out_ft = out_ft.at[:, -self.modes1:, :self.modes2, :].set(s2)

    # Go back to the spatial domain
    y = jnp.fft.irfftn(out_ft, axes=(1, 2))

    return y

class Project(nn.Module):
  in_channels: int = 32
  out_channels: int = 32
  activation: Callable = nn.gelu

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(self.in_channels)(x)
    x = self.activation(x)
    x = nn.Dense(self.out_channels)(x)
    return x


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
      (inputs.shape[-1], self.features)
    )
    kernel_i = self.param(
      "kernel_i",
      normal(stddev=1e-2, dtype=jnp.float32),
      (inputs.shape[-1], self.features)
    )
    kernel = kernel_r + 1j*kernel_i
    y = lax.dot_general(inputs, kernel,
                      (((inputs.ndim - 1,), (0,)), ((), ())))
    if self.use_bias:
      bias_r = self.param(
        "bias_r",
        normal(stddev=1e-2, dtype=jnp.float32),
        (self.features,)
      )
      bias_i = self.param(
        "bias_i",
        normal(stddev=1e-2, dtype=jnp.float32),
        (self.features,)
      )
      bias = bias_r + 1j*bias_i
      y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
    return y

class CProject(nn.Module):
  in_channels: int = 32
  out_channels: int = 32
  activation: Callable = nn.gelu

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(self.in_channels)(x)
    x = self.activation(x)
    x = nn.Dense(2*self.out_channels)(x)
    x = x[..., :self.out_channels] + 1j*x[..., self.out_channels:]
    return x

def c_gelu(x):
  return nn.gelu(x.real) + 1j*nn.gelu(x.imag)

class ComplexBornStage(nn.Module):
  out_channels: int = 32
  modes1: int = 12
  modes2: int = 12
  activation: Callable = c_gelu

  @nn.compact
  def __call__(self, x, k):
    L = CProject(16, self.out_channels, self.activation)(k)
    G = CProject(16, self.out_channels, self.activation)(k)

    F_stage = ComplexSpectralConv2d(
      out_channels=self.out_channels,
      modes1=self.modes1,
      modes2=self.modes2
    )

    u =  CDense(self.out_channels)(x)
    Wu = CDense(self.out_channels)(u)
    Hu = CDense(self.out_channels)(u)
    Ms = CDense(self.out_channels)(k)

    born_scatter = Wu - G*(Hu - F_stage(L*u + Ms))
    return self.activation(born_scatter)


class BornStage(nn.Module):
  out_channels: int = 32
  modes1: int = 12
  modes2: int = 12
  activation: Callable = nn.gelu

  @nn.compact
  def __call__(self, x, k):
    L = Project(2*self.out_channels, self.out_channels, self.activation)(k)
    G = Project(2*self.out_channels, self.out_channels, self.activation)(k)

    F_stage = SpectralConv2d(
      out_channels=self.out_channels,
      modes1=self.modes1,
      modes2=self.modes2
    )

    u =  nn.Dense(self.out_channels)(x)
    Wu = nn.Dense(self.out_channels)(u)
    Hu = nn.Dense(self.out_channels)(u)
    Ms = nn.Dense(self.out_channels)(k)

    born_scatter = Wu - G*(Hu - F_stage(L*u + Ms))
    return self.activation(born_scatter)

class CBNO(nn.Module):
  r'''
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
  '''
  modes1: int = 12
  modes2: int = 12
  width: int = 32
  depth: int = 4
  channels_last_proj: int = 128
  activation: Callable = nn.gelu
  out_channels: int = 1
  padding: int = 32 # Padding for non-periodic inputs

  @nn.compact
  def __call__(self, sos, pml, src) -> jnp.ndarray:
    sos = (1/sos)**2

    # Pad input
    if self.padding > 0:
      padval = self.param(
        'padval',
        normal(1., jnp.float32),
        (3,),
        jnp.float32
      )

      sos = jnp.pad(
        sos,
        ((0, 0), (0, self.padding), (0, self.padding), (0, 0)),
        mode='constant',
        constant_values=1.0
      )

      pml = jnp.pad(
        pml,
        ((0, 0), (0, self.padding), (0, self.padding), (0, 0)),
        mode='constant',
        constant_values=1.0
      )

      src = jnp.pad(
        src,
        ((0, 0), (0, self.padding), (0, self.padding), (0, 0)),
        mode='constant',
        constant_values=0.0
      )


    # Generate coordinate grid, and append to input channels
    grid = self.get_grid(src)
    k = jnp.concatenate([src, sos, pml, grid], axis=-1)
    x = jnp.concatenate([src, sos, pml, grid], axis=-1)
    #k = jnp.concatenate([src, sos, pml], axis=-1)
    #x = jnp.concatenate([src, sos, pml], axis=-1)

    # Lift the input to a higher dimension
    x = CProject(2*self.width,self.width)(x)
    k = Project(2*self.width,self.width)(k)

    # Apply Fourier stages, last one has no activation
    # (can't find this in the paper, but is in the original code)
    for depthnum in range(self.depth):
      activation = self.activation if depthnum < self.depth - 1 else lambda x: x
      x = ComplexBornStage(
        out_channels=self.width,
        modes1=self.modes1,
        modes2=self.modes2,
        activation=activation
      )(x, k)

    # Unpad
    if self.padding > 0:
      x = x[:, :-self.padding, :-self.padding, :]

    # Project to the output channels
    x = CDense(1)(x)
    return x

  @staticmethod
  def get_grid(x):
    x1 = jnp.linspace(0, 1, x.shape[1])
    x2 = jnp.linspace(0, 1, x.shape[2])
    x1, x2 = jnp.meshgrid(x1, x2)
    grid = jnp.expand_dims(jnp.stack([x1, x2], axis=-1), 0)
    batched_grid = jnp.repeat(grid, x.shape[0], axis=0)
    return batched_grid

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

class BNO(nn.Module):
  r'''
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
  '''
  modes1: int = 12
  modes2: int = 12
  width: int = 32
  depth: int = 4
  channels_last_proj: int = 128
  activation: Callable = nn.gelu
  out_channels: int = 1
  padding: int = 32 # Padding for non-periodic inputs

  @nn.compact
  def __call__(self, sos, pml, src) -> jnp.ndarray:

    # Pad input
    if self.padding > 0:
      padval = self.param(
        'padval',
        normal(1., jnp.float32),
        (3,),
        jnp.float32
      )

      sos = jnp.pad(
        sos,
        ((0, 0), (0, self.padding), (0, self.padding), (0, 0)),
        mode='constant',
        constant_values=1.0
      )

      pml = jnp.pad(
        pml,
        ((0, 0), (0, self.padding), (0, self.padding), (0, 0)),
        mode='constant',
        constant_values=1.0
      )

      src = jnp.pad(
        src,
        ((0, 0), (0, self.padding), (0, self.padding), (0, 0)),
        mode='constant',
        constant_values=0.0
      )


    # Generate coordinate grid, and append to input channels
    grid = self.get_grid(src)
    k = jnp.concatenate([src, sos, pml, grid], axis=-1)
    x = jnp.concatenate([src, sos, pml, grid], axis=-1)
    #k = jnp.concatenate([src, sos, pml], axis=-1)
    #x = jnp.concatenate([src, sos, pml], axis=-1)

    # Lift the input to a higher dimension
    x = Project(2*self.width,self.width)(x)
    k = Project(2*self.width,self.width)(k)



    # Apply Fourier stages, last one has no activation
    # (can't find this in the paper, but is in the original code)
    for depthnum in range(self.depth):
      activation = self.activation if depthnum < self.depth - 1 else lambda x: x
      x = BornStage(
        out_channels=self.width,
        modes1=self.modes1,
        modes2=self.modes2,
        activation=activation
      )(x, k)

    # Unpad
    if self.padding > 0:
      x = x[:, :-self.padding, :-self.padding, :]

    # Project to the output channels
    x = Project(self.channels_last_proj,self.out_channels)(x)
    return x

  @staticmethod
  def get_grid(x):
    x1 = jnp.linspace(0, 1, x.shape[1])
    x2 = jnp.linspace(0, 1, x.shape[2])
    x1, x2 = jnp.meshgrid(x1, x2)
    grid = jnp.expand_dims(jnp.stack([x1, x2], axis=-1), 0)
    batched_grid = jnp.repeat(grid, x.shape[0], axis=0)
    return batched_grid

class WrappedBNO(nn.Module):
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
    y = BNO(
      depth=self.stages,
      width=self.channels,
      out_channels=out_channels,
      padding=32,
      modes1=32,
      modes2=32,
    )(sos, pml, src)

    if self.dtype == jnp.complex64:
      return jnp.expand_dims(y[...,0] + 1j*y[...,1], -1)
    else:
      return jnp.expand_dims(jnp.sqrt(y[...,0]**2 + y[...,1]**2), -1)
