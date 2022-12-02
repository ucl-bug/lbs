import pickle
from hashlib import sha256
from typing import Tuple

import numpy as np
from jax import numpy as jnp
from torch.utils.data import Dataset
from tqdm import tqdm


class MNISTHelmholtz(Dataset):
    def __init__(
        self,
        image_size: int = 128,
        pml_size: int = 32,
        sound_speed_lims: Tuple[float] = [1.0, 1.3],
        omega: float = 1.0,
        num_samples: int = 10000,
        regenerate: bool = False,
        dtype: jnp.dtype = jnp.float32,
    ):
        r"""A dataset of Helmholtz fields with MNIST derived speeds of sound."""
        # Store the parameters
        self.image_size = image_size
        self.pml_size = pml_size
        self.sound_speed_lims = sound_speed_lims
        self.omega = omega
        self.num_samples = num_samples
        self.dtype = dtype

        # Generate the dataset
        if regenerate:
            self.generate()

        # Load the dataset
        try:
            self.load()
        except:
            print("Dataset not found, generating it")
            self.generate()
            self.load()

    @property
    def filepath(self):
        r"""Generates a filename using hashlib.
        Hoping no collisions occour.."""

        filename = f"{self.image_size}_{self.pml_size}_{self.sound_speed_lims}_{self.omega}_{self.num_samples}"
        hashed_name = sha256(filename.encode()).hexdigest()
        return f"data/{hashed_name}.npz"

    def __len__(self):
        return len(self.fields)

    def __getitem__(self, idx):
        # Take absolute value if dtype is float32
        if self.dtype == jnp.float32:
            field = jnp.abs(self.fields[idx])
        else:
            field = self.fields[idx]
        return {
            "field": field,
            "sound_speed": self.sound_speed[idx],
            "pml": self.pml,
            "source": self.sources[idx],
        }

    def save(self):
        dict_to_save = {
            "data": {
                "fields": self.fields,
                "sound_speed": self.sound_speed,
                "sources": self.sources,
                "pml": self.pml,
            },
            "image_size": self.image_size,
            "pml_size": self.pml_size,
            "sound_speed_lims": self.sound_speed_lims,
            "omega": self.omega,
            "num_samples": self.num_samples,
        }
        with open(self.filepath, "wb") as f:
            pickle.dump(dict_to_save, f)

    def load(self):
        with open(self.filepath, "rb") as f:
            dict_to_load = pickle.load(f)

        self.fields = dict_to_load["data"]["fields"]
        self.sound_speed = dict_to_load["data"]["sound_speed"]
        self.sources = dict_to_load["data"]["sources"]
        self.pml = dict_to_load["data"]["pml"]

        self.image_size = dict_to_load["image_size"]
        self.pml_size = dict_to_load["pml_size"]
        self.sound_speed_lims = dict_to_load["sound_speed_lims"]
        self.omega = dict_to_load["omega"]
        self.num_samples = dict_to_load["num_samples"]

    def generate(self):
        r"""Generates the dataset"""
        # Local imports, this function is used probably only once so it's fine
        # to import its dependencies here
        from jax import jit
        from jax.image import resize
        from jwave import FourierSeries
        from jwave.acoustics.time_harmonic import helmholtz, helmholtz_solver
        from jwave.geometry import Domain, Medium
        from torchvision.datasets import MNIST

        # Download or load the MNIST images
        print("Getting MNIST images")
        mnist_images = (
            MNIST("data/mnist_images_original", download=True)
            .data.numpy()
            .astype(float)
        )

        # Keep only self.num_samples images
        if self.num_samples < len(mnist_images):
            mnist_images = mnist_images[: self.num_samples]
        else:
            raise ValueError(
                f"num_samples must be smaller than the number of MNIST images ({len(mnist_images)})"
            )
        mnist_images = mnist_images / 255.0

        # Resize the images
        def prepare_image(image):
            # Resizing
            im = resize(image, (self.image_size, self.image_size), "nearest")
            im = jnp.pad(im, self.pml_size, mode="edge")
            im = jnp.expand_dims(im, -1)

            # Fixing range
            im = (
                im * (self.sound_speed_lims[1] - self.sound_speed_lims[0])
                + self.sound_speed_lims[0]
            )
            return im

        self.sound_speed = []
        print("Getting speeds of sound from MNIST images")
        for image in tqdm(mnist_images):
            self.sound_speed.append(prepare_image(image))

        ### Simulations

        # Defining simulation parameters
        N = tuple([self.image_size + 2 * self.pml_size] * 2)
        dx = (1.0, 1.0)
        domain = Domain(N, dx)

        # Source field
        src_field = jnp.zeros(N).astype(jnp.complex64)
        src_field = src_field.at[1, 1].set(10.0)
        src = FourierSeries(jnp.expand_dims(src_field, -1), domain) * self.omega

        # Get the pml
        medium = Medium(
            domain=domain, sound_speed=self.sound_speed_lims[1], pml_size=self.pml_size
        )
        sim_params = helmholtz.default_params(src, medium, self.omega)
        pml = sim_params["pml_on_grid"][0].on_grid
        self.pml = jnp.stack(
            [pml[..., 0].real, pml[..., 0].imag, pml[..., 1].real, pml[..., 1].imag], -1
        )
        del sim_params

        # Defining the simulation function
        @jit
        def simulate(sos, src):
            sos = FourierSeries(sos, domain)
            medium = Medium(domain=domain, sound_speed=sos, pml_size=self.pml_size)
            return helmholtz_solver(medium, self.omega, -src)

        # Local function to get the source field
        def random_src_on_background(sound_speed):
            """
            good_sample = False
            while not good_sample:
                pos = np.random.randint(
                    self.pml_size + 1, sound_speed.shape[0] - self.pml_size - 1, 2
                )
                if sound_speed[pos[0], pos[1]] <= self.sound_speed_lims[0] + 1e-5:
                    good_sample = True
            print(f"Source at {pos}")
            """
            pos = [32, 32]
            # Get source field
            N = tuple([self.image_size + 2 * self.pml_size] * 2)
            dx = (1.0, 1.0)
            domain = Domain(N, dx)

            src_field = np.zeros(N).astype(np.complex64)
            src_field[pos[0], pos[1]] = 10.0
            # src_field = smooth(src_field) + 0j
            src = FourierSeries(jnp.expand_dims(src_field, -1), domain) * self.omega
            return src

        # Running the simulations
        self.fields = []
        self.sources = []
        print("Running simulations")
        for sos in tqdm(self.sound_speed):
            src = random_src_on_background(sos)
            outfield = simulate(sos, src)

            # Crop the fields to remove pml
            field = self.crop_pml(outfield.on_grid)
            src = self.crop_pml(src.on_grid).real

            # Add the field to the dataset
            self.fields.append(field)
            self.sources.append(src)

        # Crop the fields to remove pml
        self.pml = self.crop_pml(self.pml)
        for idx, field in enumerate(self.sound_speed):
            self.sound_speed[idx] = self.crop_pml(field)

        # Saving the dataset
        print("Saving dataset")
        self.save()

    def crop_pml(self, field):
        return field[
            self.pml_size : -self.pml_size,
            self.pml_size : -self.pml_size,
        ]


def collate_fn(batch):
    return {
        "field": jnp.stack([item["field"] for item in batch], axis=0),
        "sound_speed": jnp.stack([item["sound_speed"] for item in batch], axis=0),
        "pml": jnp.stack([item["pml"] for item in batch], axis=0),
        "source": jnp.stack([item["source"] for item in batch], axis=0),
    }
