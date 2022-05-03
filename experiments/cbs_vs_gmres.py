'''
This script evaluates the accuracy of the CBS method for solving the
homogeneous Helmholtz equation in the unit square, against the
GMRES solution
'''

import argparse

import neptune.new as neptune
import numpy as np
from jax import jit
from jax import numpy as jnp
from jaxdf.discretization import FourierSeries
from jwave.acoustics.time_harmonic import helmholtz_solver
from jwave.geometry import Domain, Medium
from matplotlib import pyplot as plt

from bno.cbs import born_solver
from bno.logging import Logger


def main(args, logger):
  a0 = np.log10(args.alpha_min)
  a1 = np.log10(args.alpha_max)
  alphas = np.linspace(a0, a1, args.alpha_num)
  alphas = np.power(10, alphas)

  n0 = np.log10(args.pml_min)
  n1 = np.log10(args.pml_max)
  pmls = np.linspace(n0, n1, args.pml_num)
  pmls = np.power(10, pmls).astype(int)

  # Define unpedded source field
  domain = Domain((args.size, args.size), (1.0, 1.0))
  sos = jnp.ones(domain.N)
  src = jnp.zeros((args.size, args.size)) + 0j
  src = src.at[args.size//2,args.size//2].set(1.0 + 0j)

  src = FourierSeries(jnp.expand_dims(src, -1), domain)
  sos = FourierSeries(jnp.expand_dims(sos, -1), domain)
  # Make reference solution
  domain = Domain((args.size+64, args.size+64), (1.0, 1.0))
  src_padded = jnp.pad(src.on_grid, ((32,32), (32,32), (0,0)))
  src_padded = FourierSeries(src_padded, domain)
  medium = Medium(domain, sound_speed=1.0, pml_size=32)
  ref_field = helmholtz_solver(medium, args.omega, src_padded).on_grid[32:-32,32:-32,0]
  max_ref = jnp.amax(jnp.abs(ref_field))
  norm_ref = jnp.linalg.norm(ref_field)

  maxval = jnp.max(jnp.abs(ref_field))
  plt.imshow(jnp.real(ref_field), vmin = -maxval, vmax=maxval, cmap="seismic")
  plt.colorbar()
  plt.savefig('ref_born_real.png')
  plt.close()

  maxval = jnp.max(jnp.abs(ref_field))
  plt.imshow(jnp.imag(ref_field), vmin = -maxval, vmax=maxval, cmap="seismic")
  plt.colorbar()
  plt.savefig('ref_born_imag.png')
  plt.close()

  l_2 = np.zeros((args.alpha_num, args.pml_num))
  l_inf = np.zeros((args.alpha_num, args.pml_num))


  print(src.domain)
  print(sos.domain)

  def wrapped_solver(pml_size, alpha):
    return born_solver(
      sos,
      -src,
      omega=args.omega,
      k0=1.0,
      pml_size=pml_size,
      max_iter=10000,
      tol=-1.0,
      alpha=alpha
    )

  solver = jit(wrapped_solver, static_argnames=['pml_size'])

  for r, alpha in enumerate(alphas):
    for c, pml in enumerate(pmls):
      outfield, _ = solver(pml, alpha)

      linf = 100*np.amax(jnp.abs(outfield - ref_field))/max_ref
      l2 = 100*np.linalg.norm(outfield - ref_field)/norm_ref

      print(f"alpha={alpha}, pml={pml}, l2={l2}, linf={linf}")
      l_2[r,c] = l2
      l_inf[r,c] = linf

      # Log image
      maxval = jnp.max(jnp.abs(outfield))
      plt.imshow(jnp.real(outfield), vmin = -maxval, vmax=maxval, cmap="seismic")
      plt.colorbar()
      plt.savefig('homog_born_real.png')
      plt.close()

      maxval = jnp.max(jnp.abs(outfield))
      plt.imshow(jnp.imag(outfield), vmin = -maxval, vmax=maxval, cmap="seismic")
      plt.colorbar()
      plt.savefig('homog_born_imag.png')
      plt.close()

  # Log the error heatmaps in neptune, with colorbar
  fig, ax = plt.subplots(1, 1)
  img = ax.contourf(
    np.log10(alphas), np.log10(pmls), np.log10(l_2),
    cmap="inferno", levels=[-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
  ax.set_xlabel("Alpha num (log10)")
  ax.set_ylabel("PML num (log10)")
  ax.set_title("L2 error (log %)")
  plt.colorbar(img, ax=ax)
  logger.experiment["l2_error"].upload(neptune.types.File.as_image(fig))

  fig, ax = plt.subplots(1, 1)
  img = ax.contourf(
    np.log10(alphas), np.log10(pmls), np.log10(l_inf),
    cmap="inferno", levels=[-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
  ax.set_xlabel("Alpha num (log10)")
  ax.set_ylabel("PML num (log10)")
  ax.set_title("L-inf error (log %)")
  plt.colorbar(img, ax=ax)
  logger.experiment["linf_error"].upload(neptune.types.File.as_image(fig))


if __name__ == "__main__":
  # Parse arguments
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument('--size', type=int, default=128)
  arg_parser.add_argument('--omega', type=float, default=1.0)

  arg_parser.add_argument('--alpha_min', type=float, default=1.0)
  arg_parser.add_argument('--alpha_max', type=float, default=10.0)
  arg_parser.add_argument('--alpha_num', type=int, default=10)

  arg_parser.add_argument('--pml_min', type=int, default=8)
  arg_parser.add_argument('--pml_max', type=int, default=128)
  arg_parser.add_argument('--pml_num', type=int, default=10)

  args = arg_parser.parse_args()

  # Logging the inputs
  logger = Logger(name="cbs_accuracy")
  logger.experiment["parameters"] = vars(args)

  # Running the experiment
  main(args, logger)
