'''
This script evaluates the accuracy of the CBS method for solving the
homogeneous Helmholtz equation in the unit square, against the
GMRES solution
'''

import argparse
from time import time

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

  pmls = [4, 8, 16, 32, 64, 128, 192]

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

  print(src.domain)
  print(sos.domain)

  gt  = FourierSeries(jnp.expand_dims(ref_field, -1), src.domain)

  def wrapped_solver(pml_size, alpha):
    return born_solver(
      sos,
      -src,
      omega=args.omega,
      k0=1.0,
      pml_size=pml_size,
      max_iter=10000,
      tol=-1.0,
      alpha=alpha,
      gt = gt
    )

  solver = jit(wrapped_solver, static_argnames=['pml_size'])

  # Log the error heatmaps in neptune, with colorbar
  fig, ax = plt.subplots(1, 1)
  ax.set_prop_cycle('color',[plt.cm.copper(i) for i in np.linspace(0, 1, len(alphas))])

  for r, alpha in enumerate(alphas):
    exec_time = []
    num_pml_pts = []
    for c, pml in enumerate(pmls):
      outfield, _ = solver(pml, alpha)

      linf = 100*np.amax(jnp.abs(outfield - ref_field))/max_ref

      if linf < 1.0:
        # Time and repeat
        start_time = time()
        outfield = solver(pml, alpha)[0].block_until_ready()
        exec_time.append(time() - start_time)
        num_pml_pts.append(pml)

        print(f"PML: {pml}, alpha: {alpha}, Execution time: {exec_time[-1]}")
      else:
        print(f"PML: {pml}, alpha: {alpha}, Execution time: NaN")

    # Add line to the plot
    ax.plot(num_pml_pts, exec_time, label=f"alpha={alpha}")

  # Log the plot to neptune with legend
  ax.legend()
  ax.set_yscale('log')
  ax.set_xlabel('PML size')
  ax.set_ylabel('Execution time (s)')
  ax.set_title(f"Execution time vs. PML size for different alphas")
  logger.experiment["exec_time"].upload(neptune.types.File.as_image(fig))


if __name__ == "__main__":
  # Parse arguments
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument('--size', type=int, default=128)
  arg_parser.add_argument('--omega', type=float, default=1.0)

  arg_parser.add_argument('--alpha_min', type=float, default=1.0)
  arg_parser.add_argument('--alpha_max', type=float, default=10.0)
  arg_parser.add_argument('--alpha_num', type=int, default=10)

  args = arg_parser.parse_args()

  # Logging the inputs
  logger = Logger(name="cbs_accuracy")
  logger.experiment["parameters"] = vars(args)

  # Running the experiment
  main(args, logger)
