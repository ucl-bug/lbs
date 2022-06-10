import argparse
from functools import partial

import flax.linen as nn
import numpy as np
import optax
from jax import jit
from jax import numpy as jnp
from jax import random, value_and_grad
from matplotlib import pyplot as plt
from torch import Generator
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from bno.datasets import MNISTHelmholtz, collate_fn
from bno.modules import WrappedBNO, WrappedComplexBNO, WrappedFNO, WrappedUBS

RNG = random.PRNGKey(0)

def print_config(d):
  print("--- Config ---")
  for k, v in d.items():
    print("{:<35} {:<20}".format(k, str(v)))
  print("--- End Config ---\n")

def log_wandb_image(wandb, name, step, sos, field, pred_field):
  fig, ax = plt.subplots(1, 3, figsize=(12, 4))

  ax[0].imshow(sos, cmap="inferno")
  ax[0].set_title("Sound speed")

  ax[1].imshow(field.real, vmin=-5, vmax=5, cmap="RdBu_r")
  ax[1].set_title("Field")

  ax[2].imshow(pred_field.real, vmin=-5, vmax=5, cmap="RdBu_r")
  ax[2].set_title("Predicted field")

  #plt.show()

  img = wandb.Image(plt)
  wandb.log({name: img}, step=step)
  plt.close()

def log_with_intermediates(wandb, step, sos, field, pred_field, intermediates):
  # Extracting intermediates
  fields = [x[0] for x in intermediates['fields']]
  M1 = [x['M1'][0] for x in intermediates['operators']]
  M2 = [x['M2'][0] for x in intermediates['operators']]
  src = [x['src'][0] for x in intermediates['operators']]
  updates = [x[0] for x in intermediates['updates']]
  num_figures = max([2, len(fields)])

  # Log in rows
  fig, ax = plt.subplots(num_figures, 9, figsize=(24, num_figures*3))
  for i in range(len(fields)):
    maxval = np.amax(jnp.abs(fields[i])).item()
    ax[i, 0].imshow(fields[i].real, vmin=-maxval/2, vmax=maxval/2, cmap="seismic")
    ax[i, 0].set_title("Field (real)")
    ax[i, 1].imshow(fields[i].imag, vmin=-maxval/2, vmax=maxval/2, cmap="seismic")
    ax[i, 1].set_title("Field (imag)")

    if i < len(M1):
      maxval = np.amax(jnp.abs(M1[i])).item()
      ax[i, 2].imshow(M1[i].real, vmin=-maxval, vmax=maxval, cmap="seismic")
      ax[i, 2].set_title("M1 (real)")
      ax[i, 3].imshow(M1[i].imag, vmin=-maxval, vmax=maxval, cmap="seismic")
      ax[i, 3].set_title("M1 (imag)")

      maxval = np.amax(jnp.abs(M2[i])).item()
      ax[i, 4].imshow(M2[i].real, vmin=-maxval, vmax=maxval, cmap="seismic")
      ax[i, 4].set_title("M2 (real)")
      ax[i, 5].imshow(M2[i].imag, vmin=-maxval, vmax=maxval, cmap="seismic")
      ax[i, 5].set_title("M2 (imag)")

      maxval = np.amax(jnp.abs(src[i])).item()
      ax[i, 6].imshow(src[i].real, vmin=-maxval, vmax=maxval, cmap="seismic")
      ax[i, 6].set_title("Src (real)")
      ax[i, 7].imshow(src[i].imag, vmin=-maxval, vmax=maxval, cmap="seismic")
      ax[i, 7].set_title("Src (imag)")

    maxval = np.amax(jnp.abs(updates[i])).item()
    ax[i, 8].imshow(jnp.abs(updates[i]), vmin=-0, vmax=maxval, cmap="inferno")
    ax[i, 8].set_title("Update magnitude (& next field)")

  img = wandb.Image(plt)
  wandb.log({"intermediates": img}, step=step)
  plt.close()


def main(args):
  # Check arguments
  assert args.max_sos > 1.0, "max_sos must be greater than 1.0"
  assert args.model in ["fno", "bno", 'cbno', 'ubs'], "model must be 'fno'"
  assert args.batch_size > 0, "batch_size must be greater than 0"
  assert args.stages > 0, "stages must be greater than 0"
  assert args.channels > 0, "channels must be greater than 0"
  assert args.target in ['amplitude', 'complex'], "target must be 'amplitude' or 'complex'"

  args.target = jnp.complex64 if args.target == "complex" else jnp.float32

  # Print arguments nicely
  print_config(vars(args))

  # Load dataset
  print("Loading dataset...")
  dataset = MNISTHelmholtz(
    image_size=128,
    pml_size=16,
    sound_speed_lims=[1., args.max_sos],
    source_pos=(8+16, 8+16),  # In pixels
    omega=1.0,
    num_samples=1500,
    regenerate=False,
    dtype=args.target,
    )

  # Splitting dataset
  train_size = int(0.8 * len(dataset))
  val_size = int(0.1 * len(dataset))
  test_size = len(dataset) - train_size - val_size

  trainset, valset, testset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=Generator().manual_seed(0)
  )

  # Making dataloaders
  trainloader = DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
  )
  validloader = DataLoader(
    valset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
    drop_last=True
  )

  # Initialize model
  print("Setting up model...")
  if args.model == "fno":
    model = WrappedFNO(
      stages=args.stages,
      channels=args.channels,
      dtype= args.target
    )
  elif args.model == "bno":
    model = WrappedBNO(
      stages=args.stages,
      channels=args.channels,
      dtype= args.target
    )
  elif args.model == "cbno":
    model = WrappedComplexBNO(
      stages=args.stages,
      channels=args.channels,
      dtype= args.target
    )
  elif args.model == "ubs":
    model = WrappedUBS(
      stages=args.stages,
      dtype= args.target,
    )

  _sos = jnp.ones((1, dataset.image_size, dataset.image_size, 1))
  _pml = jnp.ones((1, dataset.image_size, dataset.image_size, 4))
  _src = jnp.ones((1, dataset.image_size, dataset.image_size, 1))
  model_params = model.init(
    RNG, _sos, _pml, _src, unrolls=args.stages
  )
  del _sos
  del _pml
  del _src

  # Initialize optimizer
  optimizer = optax.adamw(learning_rate=args.lr)
  opt_state = optimizer.init(model_params)

  # Define loss
  @partial(jit, static_argnums=5)
  def loss(model_params, sound_speed, field, pml, src, unrolls):
    # Predict fields
    pred_field = model.apply(model_params, sound_speed, pml, src, unrolls)

    # Compute loss
    lossval = jnp.mean(jnp.abs(pred_field - field)**2)
    return lossval

  @partial(jit, static_argnums=4)
  def predict(model_params, sound_speed, pml, src, unrolls):
    return model.apply(model_params, sound_speed, pml, src, unrolls)

  @partial(jit, static_argnums=4)
  def predict_with_intermediate(model_params, sound_speed, pml, src, unrolls):
    def _fun(model):
      return model.apply_with_intermediate(sound_speed, pml, src, unrolls)
    return nn.apply(_fun, model)(model_params)

  @partial(jit, static_argnums=3)
  def update(opt_state, params, batch, unrolls):
    # Get loss and gradients
    lossval, gradients = value_and_grad(loss)(
      params,
      batch["sound_speed"],
      batch["field"],
      batch["pml"],
      batch["source"],
      unrolls
    )

    updates, opt_state = optimizer.update(gradients, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, lossval

  # Initialize wandb
  print("Training...")
  wandb.init('bno')
  wandb.config.update(args)

  # Training loop
  step = 0
  old_loss = 1e20
  #  Take a checkpoint of the model params
  params_ckpt = [model_params.copy({}), model_params.copy({})]
  optstate_ckpt = [opt_state, opt_state]

  for epoch in range(args.epochs):
    unrolls = args.stages  # 1 + int(epoch / 30)
    print(f"Epoch {epoch}, unrolls {unrolls}")


    with tqdm(trainloader, unit="batch") as tepoch:
      for batch in tepoch:
        tepoch.set_description(f"Epoch {epoch}")

        # Update parameters
        model_params, opt_state, lossval = update(
          opt_state, model_params, batch, unrolls
        )

        # Check if loss exploded, in which case we restore the model params
        #if lossval > 10*old_loss:
        #  print("Training exploded, restoring model params of previous 5 steps")
        #  model_params = params_ckpt[0]
        #  opt_state = optstate_ckpt[0]
        #elif step % 50 == 0:
        #  old_loss = lossval
        #  params_ckpt = [params_ckpt[1]] + [model_params.copy({})]
        #  optstate_ckpt = [optstate_ckpt[1]] + [opt_state]

        # Log to wandb
        wandb.log({"loss": lossval}, step=step)

        # Update progress bar
        tepoch.set_postfix(loss=lossval)

        # Update step
        step += 1

    # Log training image
    if epoch % 5 == 0:
      sos = jnp.expand_dims(batch["sound_speed"][0], axis=0)
      pml = jnp.expand_dims(batch["pml"][0], axis=0)
      src = jnp.expand_dims(batch["source"][0], axis=0)
      field = batch["field"][0]

      pred_fields, intermediates = predict_with_intermediate(model_params, sos, pml, src, unrolls)
      pred_field = pred_fields[0]
      sos = sos[0]

      log_with_intermediates(
        wandb, step, sos, field, pred_field, intermediates)
      log_wandb_image(wandb, "training", step, sos, field, pred_field)

    # Validation
    avg_loss = 0
    val_steps = 0
    with tqdm(validloader, unit="batch") as tval:
      for batch in tval:
        tval.set_description(f"Epoch (val) {epoch}")

        lossval = loss(
          model_params,
          batch["sound_speed"],
          batch["field"],
          batch["pml"],
          batch["source"],
          unrolls
        )
        avg_loss += lossval*len(batch["sound_speed"])
        tval.set_postfix(loss=lossval)
        val_steps += len(batch["sound_speed"])

    wandb.log({"val_loss": avg_loss/val_steps}, step=step)

    # Log validation image
    if epoch % 5 == 0:
      sos = jnp.expand_dims(batch["sound_speed"][0], axis=0)
      pml = jnp.expand_dims(batch["pml"][0], axis=0)
      src = jnp.expand_dims(batch["source"][0], axis=0)
      field = batch["field"][0]
      pred_field = predict(model_params, sos, pml, src, unrolls)[0]
      sos = sos[0]
      log_wandb_image(wandb, "validation", step, sos, field, pred_field)


if __name__ == '__main__':
  # Parse arguments
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument('--max_sos', type=float, default=2.0)
  arg_parser.add_argument('--model', type=str, default='ubs')
  arg_parser.add_argument('--batch_size', type=int, default=16)
  arg_parser.add_argument('--epochs', type=int, default=1000)
  arg_parser.add_argument('--lr', type=float, default=1e-3)
  arg_parser.add_argument('--stages', type=int, default=6)
  arg_parser.add_argument('--channels', type=int, default=32)
  arg_parser.add_argument('--target', type=str, default='complex')

  args = arg_parser.parse_args()

  # Start training
  main(args)
