import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float  # https://github.com/google/jaxtyping
 

class CNN(eqx.Module):
    layers: list
    
    def __init__(self, key):
        key1, key2, key3, key4 = jax.random.split(key,4)

        self.layers = [
            eqx.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=4, key=key1),
            eqx.nn.MaxPool2d(kernel_size=2),
            jax.nn.relu,
            jnp.ravel,
            eqx.nn.Linear(in_features=1728, out_features=512, key=key2),
            jax.nn.sigmoid,
            eqx.nn.Linear(in_features=512, out_features=64, key=key3),
            jax.nn.relu,
            eqx.nn.Linear(in_features=64, out_features=10, key=key4),
            jax.nn.log_softmax
        ]

    def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "10"]: # forward pass
        for layer in self.layers:
            x=layer(x)
        return x

 