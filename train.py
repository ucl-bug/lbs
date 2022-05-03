import argparse

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
from bno.models import WrappedBNO, WrappedFNO

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

  ax[1].imshow(field.real, vmin=-1, vmax=1, cmap="RdBu_r")
  ax[1].set_title("Field")

  ax[2].imshow(pred_field.real, vmin=-1, vmax=1, cmap="RdBu_r")
  ax[2].set_title("Predicted field")

  #plt.show()

  img = wandb.Image(plt)
  wandb.log({name: img}, step=step)
  plt.close()


def main(args):
  # Check arguments
  assert args.max_sos > 1.0, "max_sos must be greater than 1.0"
  assert args.model in ["fno", "bno"], "model must be 'fno'"
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

  _sos = jnp.ones((1, dataset.image_size, dataset.image_size, 1))
  _pml = jnp.ones((1, dataset.image_size, dataset.image_size, 4))
  _src = jnp.ones((1, dataset.image_size, dataset.image_size, 1))
  output, model_params = model.init_with_output(RNG, _sos, _pml, _src)
  del _sos
  del _pml
  del _src

  # Initialize optimizer
  optimizer = optax.adamw(learning_rate=args.lr)
  opt_state = optimizer.init(model_params)

  # Define loss
  @jit
  def loss(model_params, sound_speed, field, pml, src):
    # Predict fields
    pred_field = model.apply(model_params, sound_speed, pml, src)

    # Compute loss
    lossval = jnp.mean(jnp.abs(pred_field - field)**2)
    return lossval

  @jit
  def predict(model_params, sound_speed, pml, src):
    return model.apply(model_params, sound_speed, pml, src)

  @jit
  def update(opt_state, params, batch):
    # Get loss and gradients
    lossval, gradients = value_and_grad(loss)(
      params,
      batch["sound_speed"],
      batch["field"],
      batch["pml"],
      batch["source"]
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
  for epoch in range(args.epochs):
    print(f"Epoch {epoch}")

    with tqdm(trainloader, unit="batch") as tepoch:
      for batch in tepoch:
        tepoch.set_description(f"Epoch {epoch}")

        # Update parameters
        model_params, opt_state, lossval = update(
          opt_state, model_params, batch
        )

        # Log to wandb
        wandb.log({"loss": lossval}, step=step)

        # Update progress bar
        tepoch.set_postfix(loss=lossval)

        # Update step
        step += 1

    # Log training image
    sos = jnp.expand_dims(batch["sound_speed"][0], axis=0)
    pml = jnp.expand_dims(batch["pml"][0], axis=0)
    src = jnp.expand_dims(batch["source"][0], axis=0)
    field = batch["field"][0]

    pred_field = predict(model_params, sos, pml, src)[0]
    sos = sos[0]
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
          batch["source"]
        )
        avg_loss += lossval*len(batch["sound_speed"])
        tval.set_postfix(loss=lossval)
        val_steps += len(batch["sound_speed"])

    wandb.log({"val_loss": avg_loss/val_steps}, step=step)

    # Log validation image
    sos = jnp.expand_dims(batch["sound_speed"][0], axis=0)
    pml = jnp.expand_dims(batch["pml"][0], axis=0)
    src = jnp.expand_dims(batch["source"][0], axis=0)
    field = batch["field"][0]
    pred_field = predict(model_params, sos, pml, src)[0]
    sos = sos[0]
    log_wandb_image(wandb, "validation", step, sos, field, pred_field)


if __name__ == '__main__':
  # Parse arguments
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument('--max_sos', type=float, default=2.0)
  arg_parser.add_argument('--model', type=str, default='fno')
  arg_parser.add_argument('--batch_size', type=int, default=16)
  arg_parser.add_argument('--epochs', type=int, default=1000)
  arg_parser.add_argument('--lr', type=float, default=1e-4)
  arg_parser.add_argument('--stages', type=int, default=6)
  arg_parser.add_argument('--channels', type=int, default=32)
  arg_parser.add_argument('--target', type=str, default='complex')

  args = arg_parser.parse_args()

  # Start training
  main(args)
