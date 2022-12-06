# Learned Born Series

This repository contains the code for the paper

> [A Learned Born Series for Highly-Scattering Media]()

This work presents a method for solving the Helmholtz differential equation using a deep learning approach. We propose a modification to the existing convolutional Born series method to reduce the number of iterations required to solve the equation in highly-scattering media. This is achieved by transforming the linear operator into a non-linear one using a deep learning model. The method is tested on simulated examples, showing improved convergence compared to the original convolutional Born series method.

This repository can also be installed as a Python package using `pip`, to provide an implementation of the method in the [Flax neural network library](https://github.com/google/flax), as well as a Flax implementation of the Convergent Born Series by [Osnabrugge et al., 2016](https://www.sciencedirect.com/science/article/pii/S0021999116302595).

<br/>

## Installation

To install the package, clone the repository and run

```bash
pip install -e .
```

This will install the package in editable mode, so that any changes to the code will be reflected in the installed package.
