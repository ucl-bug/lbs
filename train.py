import fire
import numpy as np
import optax
from addict import Dict
from flax.training import checkpoints
from jax import jit
from jax import numpy as jnp
from jax import random, value_and_grad
from matplotlib import pyplot as plt
from torch import Generator
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from bno.datasets import MNISTHelmholtz, collate_fn
from bno.modules import (
    WrappedBNO,
    WrappedBNOS,
    WrappedCBNO,
    WrappedFNO,
    WrappedLBS,
)

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

    ax[1].imshow(field.real, vmin=-0.5, vmax=0.5, cmap="RdBu_r")
    ax[1].set_title("Field")

    ax[2].imshow(pred_field.real, vmin=-5, vmax=5, cmap="RdBu_r")
    ax[2].set_title("Predicted field")

    # plt.show()

    img = wandb.Image(plt)
    wandb.log({name: img}, step=step)
    plt.close()


def log_with_intermediates(wandb, step, sos, field, pred_field, intermediates):
    # Extracting intermediates
    fields = [x[0] for x in intermediates["fields"]]
    M1 = [x["M1"][0] for x in intermediates["operators"]]
    M2 = [x["M2"][0] for x in intermediates["operators"]]
    src = [x["src"][0] for x in intermediates["operators"]]
    updates = [x[0] for x in intermediates["updates"]]
    num_figures = max([2, len(fields)])

    # Log in rows
    fig, ax = plt.subplots(num_figures, 9, figsize=(24, num_figures * 3))
    for i in range(len(fields)):
        maxval = np.amax(jnp.abs(fields[i])).item()
        ax[i, 0].imshow(fields[i].real, vmin=-0.5, vmax=0.5, cmap="RdBu_r")
        ax[i, 1].imshow(fields[i].imag, vmin=-0.5, vmax=0.5, cmap="RdBu_r")

        if i == 0:
            ax[i, 0].set_title("Field (real)")
            ax[i, 1].set_title("Field (imag)")
            ax[i, 2].set_title("M1 (real)")
            ax[i, 3].set_title("M1 (imag)")
            ax[i, 4].set_title("M2 (real)")
            ax[i, 5].set_title("M2 (imag)")
            ax[i, 6].set_title("M3 (real)")
            ax[i, 7].set_title("M3 (imag)")
            ax[i, 8].set_title("Update magnitude (& next field)")

        # Turn off all axes
        ax[i, 0].axis("off")
        ax[i, 1].axis("off")
        ax[i, 2].axis("off")
        ax[i, 3].axis("off")
        ax[i, 4].axis("off")
        ax[i, 5].axis("off")
        ax[i, 6].axis("off")
        ax[i, 7].axis("off")
        ax[i, 8].axis("off")

        if i < len(M1):
            maxval = np.amax(jnp.abs(M1[i])).item()
            ax[i, 2].imshow(M1[i].real, vmin=-maxval, vmax=maxval, cmap="RdBu_r")
            ax[i, 3].imshow(M1[i].imag, vmin=-maxval, vmax=maxval, cmap="RdBu_r")

            maxval = np.amax(jnp.abs(M2[i])).item()
            ax[i, 4].imshow(M2[i].real, vmin=-maxval, vmax=maxval, cmap="RdBu_r")
            ax[i, 5].imshow(M2[i].imag, vmin=-maxval, vmax=maxval, cmap="RdBu_r")

            maxval = np.amax(jnp.abs(src[i])).item()
            ax[i, 6].imshow(src[i].real, vmin=-maxval, vmax=maxval, cmap="RdBu_r")
            ax[i, 7].imshow(src[i].imag, vmin=-maxval, vmax=maxval, cmap="RdBu_r")

        maxval = np.amax(jnp.abs(updates[i])).item()
        ax[i, 8].imshow(jnp.abs(updates[i]), vmin=-0, vmax=maxval, cmap="inferno")

    img = wandb.Image(plt)
    wandb.log({"intermediates": img}, step=step)
    plt.close()


def parse_args(args):
    args = Dict(args)

    # Check arguments
    assert args.max_sos > 1.0, "max_sos must be greater than 1.0"
    assert args.model in [
        "fno",
        "bno",
        "lbs",
        "cbno",
        "bno_series",
    ], "model must be 'fno'"
    assert args.batch_size > 0, "batch_size must be greater than 0"
    assert args.stages > 0, "stages must be greater than 0"
    assert args.channels > 0, "channels must be greater than 0"
    assert args.target in [
        "amplitude",
        "complex",
    ], "target must be 'amplitude' or 'complex'"

    # Add target
    args.target = jnp.complex64 if args.target == "complex" else jnp.float32

    # Print arguments nicely
    print_config(args)

    return args


def make_datasets(args):
    print("Loading dataset...")
    dataset = MNISTHelmholtz(
        image_size=128,
        pml_size=16,
        sound_speed_lims=[1.0, args.max_sos],
        omega=1.0,
        num_samples=2000,
        regenerate=False,
        dtype=args.target,
    )

    # Splitting dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    trainset, valset, testset = random_split(
        dataset, [train_size, val_size, test_size], generator=Generator().manual_seed(0)
    )

    # Making dataloaders
    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    validloader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    return trainloader, validloader, dataset.image_size


def main(
    batch_size=16,
    channels=32,
    epochs=1000,
    last_projection_channels=128,
    lr=1e-3,
    max_sos=2.0,
    model="fno",
    stages=6,
    target="complex",
):

    # Collect arguments into addict.Dict
    args = {
        "batch_size": batch_size,
        "channels": channels,
        "epochs": epochs,
        "last_projection_channels": last_projection_channels,
        "lr": lr,
        "max_sos": max_sos,
        "model": model,
        "stages": stages,
        "target": target,
    }
    args = parse_args(args)

    # Load dataset
    trainloader, validloader, image_size = make_datasets(args)

    # Initialize model
    print("Setting up model...")
    if args.model == "fno":
        model = WrappedFNO(
            stages=args.stages, channels=args.channels, dtype=args.target
        )
    elif args.model == "bno":
        model = WrappedBNO(
            stages=args.stages, channels=args.channels, dtype=args.target
        )
    elif args.model == "lbs":
        model = WrappedLBS(
            stages=args.stages, channels=args.channels, dtype=args.target
        )
    elif args.model == "cbno":
        model = WrappedCBNO(
            stages=args.stages, channels=args.channels, dtype=args.target
        )
    elif args.model == "bno_series":
        model = WrappedBNOS(
            stages=args.stages, channels=args.channels, dtype=args.target
        )
    else:
        raise NotImplementedError(f"Model {args.model} not implemented")

    _sos = jnp.ones((1, image_size, image_size, 1))
    _pml = jnp.ones((1, image_size, image_size, 4))
    _src = jnp.ones((1, image_size, image_size, 1))
    model_params = model.init(RNG, _sos, _pml, _src)

    # Test model
    print("Testing model...")
    output = model.apply(model_params, _sos, _pml, _src)
    print("Output shape:", output.shape)
    print("Output type:", output.dtype)

    del _sos
    del _pml
    del _src

    # Initialize optimizer
    """
    schedule = optax.cosine_onecycle_schedule(
        100000,
        args.lr,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=100.0
    )
    """
    schedule = args.lr

    optimizer = optax.chain(
        optax.adaptive_grad_clip(1.0),
        optax.adamw(learning_rate=schedule),
    )
    opt_state = optimizer.init(model_params)

    # Define loss
    @jit
    def loss(
        model_params,
        sound_speed,
        field,
        pml,
        src,
    ):
        # Predict fields
        pred_field = model.apply(
            model_params,
            sound_speed,
            pml,
            src,
        )

        # Compute loss
        lossval = jnp.mean(jnp.abs(pred_field - 10 * field) ** 2)
        # lossval = jnp.mean(jnp.amax(jnp.abs(pred_field - field), axis=(1,2)))
        return lossval

    @jit
    def predict(
        model_params,
        sound_speed,
        pml,
        src,
    ):
        return model.apply(model_params, sound_speed, pml, src)

    @jit
    def update(opt_state, params, batch):
        # Get loss and gradients
        lossval, gradients = value_and_grad(loss)(
            params,
            batch["sound_speed"],
            batch["field"],
            batch["pml"],
            batch["source"],
        )

        updates, opt_state = optimizer.update(gradients, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, lossval

    # Initialize wandb
    print("Training...")
    wandb.init("bno")
    wandb.config.update(args)
    run_name = wandb.run.name

    # Training loop
    step = 0
    old_v_loss = 1e100
    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")

        with tqdm(trainloader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                # Update parameters
                model_params, opt_state, lossval = update(
                    opt_state,
                    model_params,
                    batch,
                )

                # Log to wandb
                wandb.log({"loss": lossval}, step=step)

                # Update progress bar
                tepoch.set_postfix(loss=lossval)

                # Update step
                step += 1

        # Log training image
        if True:  # epoch % 5 == 0:
            sos = jnp.expand_dims(batch["sound_speed"][0], axis=0)
            pml = jnp.expand_dims(batch["pml"][0], axis=0)
            src = jnp.expand_dims(batch["source"][0], axis=0)
            field = batch["field"][0]
            pred_field = predict(
                model_params,
                sos,
                pml,
                src,
            )[0]
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
                    batch["source"],
                )
                avg_loss += lossval * len(batch["sound_speed"])
                tval.set_postfix(loss=lossval)
                val_steps += len(batch["sound_speed"])

        v_loss = avg_loss / val_steps
        wandb.log({"val_loss": v_loss}, step=step)

        # Log validation image
        if True:  # epoch % 5 == 0:
            sos = jnp.expand_dims(batch["sound_speed"][0], axis=0)
            pml = jnp.expand_dims(batch["pml"][0], axis=0)
            src = jnp.expand_dims(batch["source"][0], axis=0)
            field = batch["field"][0]
            pred_field = predict(
                model_params,
                sos,
                pml,
                src,
            )[0]
            sos = sos[0]
            log_wandb_image(wandb, "validation", step, sos, field, pred_field)

        # If the validation loss is lower, save
        if v_loss < old_v_loss:
            old_v_loss = v_loss
            print("Saving checkpoint")
            checkpoints.save_checkpoint(
                ckpt_dir=f"ckpts/{run_name}", target=opt_state, step=step
            )


if __name__ == "__main__":
    fire.Fire(main)
