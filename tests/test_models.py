from jax import numpy as jnp
from jax.random import PRNGKey

from bno.models import FNO


def test_fno():
    rng = PRNGKey(0)
    inputs = jnp.ones((8, 64, 64, 1))
    output, variables = FNO().init_with_output(rng, inputs)
    print(output.shape)
    print(output.dtype)

    # Test forward pass
    output = FNO().apply(variables, inputs)


if __name__ == "__main__":
    test_fno()
