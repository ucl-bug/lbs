# Learned Born Series

This repository contains the code for the paper

> [A Learned Born Series for Highly-Scattering Media]()

This work presents a method for solving the Helmholtz differential equation using a deep learning approach. We propose a modification to the existing convolutional Born series method to reduce the number of iterations required to solve the equation in highly-scattering media. This is achieved by transforming the linear operator into a non-linear one using a deep learning model. The method is tested on simulated examples, showing improved convergence compared to the original convolutional Born series method.

This repository can also be installed as a Python package using `pip`, to provide an implementation of the method in the [Flax neural network library](https://github.com/google/flax), as well as a Flax implementation of the Convergent Born Series by [Osnabrugge et al., 2016](https://www.sciencedirect.com/science/article/pii/S0021999116302595).

<br/>

## Installation

To install the package, clone the repository and run

```bash
pip install -r requirements.txt
pip install -e .
```

This will install the package in editable mode, so that any changes to the code will be reflected in the installed package. From here, you have a `Flax` model of the `bno`. Anywhere you can write

```python
from bno import BNO, WrappedBNO
```

and use it as a model/layer in your code. The `WrappedBNO` is made specifically for acoustic simulations, and takes care of transforming the output into a complex field.

## Train

To train the network, run

```bash
python train.py --model bno
```

Training takes about 3/4 days to complete on a single GPU, but you get good results already after a few hours.
There are several other arguments that can be passed to the script, which can be found by running

```bash
python train.py --help
```

## Test

To test a network, modify the `TRAIN_IDS` variable with your run. The key is an arbitrary string, say `my_model`, while the value needs to be the run ID of the `wandb` run. Then run

```bash
python test.py --train_id my_model
```

To generate the figures from the paper, run

```bash
python make_figures --figure example --model my_model
```

where `--figure` can be one of `example`, `iterations_error`, `show_iterations`, `show_pareto`, and `--model`.
