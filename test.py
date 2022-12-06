import os

import numpy as np
from flax import serialization
from jax import jit
from jax import numpy as jnp
from jax import random
from scipy.io import savemat
from torch import Generator
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from bno.datasets import MNISTHelmholtz, collate_fn
from bno.modules import WrappedBNO, WrappedCBS, WrappedFNO

RNG = random.PRNGKey(0)
api = wandb.Api()


def make_test_dataset(args):
    dataset = MNISTHelmholtz(
        image_size=128,
        pml_size=16,
        sound_speed_lims=[1.0, 2.0],  # [1.0, args.max_sos],
        omega=1.0,
        num_samples=2000,
        regenerate=False,
        dtype=args.target,
    )

    # Splitting dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    _, _, testset = random_split(
        dataset, [train_size, val_size, test_size], generator=Generator().manual_seed(0)
    )

    trainloader = DataLoader(
        testset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    return trainloader, dataset.image_size


def print_config(d):
    print("--- Config ---")
    for k, v in d.items():
        print("{:<35} {:<20}".format(k, str(v)))
    print("--- End Config ---\n")


def load_args(run_id: str):
    # Load the config file from the wandb run, which is
    # stored online
    run = api.run(f"bug_ucl/bno/{run_id}")
    run_name = run.name
    config = run.config
    return config, run_name


class objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


def get_ckpt_path(wandb_name):
    # Strip the integer number xxx-yyy-zzz-number
    run_number = wandb_name.split("-")[-1]

    # Find the folder in the '/ckpts/' directory that
    # ends with the run number
    folder = None
    for f in os.listdir("ckpts"):
        if f.endswith(run_number):
            folder = f
            break

    # Get the full path to the file in the folder (name not known)
    path = os.path.join("ckpts", folder)
    # Get the name of the file
    file = os.listdir(path)[0]
    # Get the full path to the file
    path = os.path.join(path, file)
    return path


def main(run_id: str):

    # If the run_id starts with "born_series", then we
    # extract the pattern "born_series_{maxiter}" from the run_id
    # and use that to set the stages parameter
    if run_id.startswith("born_series"):
        args = objdict()
        args.model = "born_series"
        args.stages = int(run_id.split("_")[-1])
        args.target = "complex"
    else:
        # Load the config file from the wandb run
        args, run_name = load_args(run_id)
        ckpt_path = get_ckpt_path(run_name)
        args = objdict(args)
        args.dtype = jnp.complex64

    # Dictionary to dataclass

    # Load test_dataset dataset
    test_loader, image_size = make_test_dataset(args)

    # Initialize model
    print("Setting up model...")
    if args.model == "fno":
        model = WrappedFNO(
            stages=args.stages, channels=args.channels, dtype=args.target
        )
    elif args.model == "bno":
        model = WrappedBNO(
            stages=args.stages,
            channels=args.channels,
            dtype=args.target,
            last_proj=args.last_projection_channels,
            use_nonlinearity=args.use_nonlinearity,
            use_grid=args.use_grid,
        )
    elif args.model == "born_series":
        model = WrappedCBS(stages=args.stages)
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

    # Load serialized parameters
    if args.model != "born_series":
        print("Loading model parameters...")
        with open(ckpt_path, "rb") as f:
            serialized_params = f.read()

        # Deserialize the parameters
        model_params = serialization.from_bytes(model_params, serialized_params)

    # Define inference function
    @jit
    def predict(model_params, sound_speed, pml, src):
        return model.apply(model_params, sound_speed, pml, src)

    # Generate test set results
    print("Generating test results...")

    results = {"true_field": [], "sound_speed": [], "source": [], "prediction": []}
    print(len(test_loader))
    with tqdm(test_loader) as tepoch:
        for sample in tepoch:
            # unpack
            sos = sample["sound_speed"]
            pml = sample["pml"]
            source = sample["source"]
            true_field = sample["field"]

            prediction = predict(model_params, sos, pml, source)

            # Update results
            results["true_field"].append(true_field[0])
            results["sound_speed"].append(sos[0])
            results["source"].append(source[0])
            results["prediction"].append(prediction[0])

    # Make them into arrays
    results["true_field"] = np.stack(results["true_field"], 0)
    results["sound_speed"] = np.stack(results["sound_speed"], 0)
    results["source"] = np.stack(results["source"], 0)
    results["prediction"] = np.stack(results["prediction"], 0)

    # Save results
    savemat(f"results/{run_id}.mat", results)


TRAIN_IDS = {
    "6_stages": "1cswiynp",
    "2_channels": "36s738nh",
    "base": "3hlvxjiq",
    "32_last_projection": "16pgi2my",
    "24_stages": "338d1jjy",
    "linear": "frep1sv4",
    "born_series_6": "born_series_6",
    "born_series_12": "born_series_12",
    "born_series_24": "born_series_24",
    "born_series_48": "born_series_48",
    "born_series_96": "born_series_96",
    "born_series_192": "born_series_192",
}

if __name__ == "__main__":
    # fire.Fire(main)

    # Test all models
    for key, value in TRAIN_IDS.items():
        if "born_series" not in key:
            print(f"Testing {key} model...")
            main(value)
