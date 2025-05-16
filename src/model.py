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
            jax.nn.relu,
            eqx.nn.MaxPool2d(kernel_size=2),
            jnp.ravel,
            eqx.nn.Linear(in_features=1728, out_features=512, key=key2),
            jax.nn.relu,
            eqx.nn.Linear(in_features=512, out_features=64, key=key3),
            jax.nn.relu,
            eqx.nn.Linear(in_features=64, out_features=10, key=key4),
            jax.nn.log_softmax
        ]

    def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "10"]: # forward pass
        for layer in self.layers:
            x=layer(x)
        return x


class CNN2(eqx.Module):
    layers: list
    
    def __init__(self, key):
        key1, key2, key3, key4, key5 = jax.random.split(key,5)

        self.layers = [
            eqx.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, key=key1),  # (1,28,28) → (8,26,26)
            jax.nn.relu,
            eqx.nn.MaxPool2d(kernel_size=2, stride=2),                                        # (8,26,26) → (8,13,13)
            eqx.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, key=key2), # (8,13,13) → (16,11,11)
            jax.nn.relu,
            eqx.nn.MaxPool2d(kernel_size=2, stride=2),                                        # (16,11,11) → (16,5,5)
            jnp.ravel,
            eqx.nn.Linear(in_features=16 * 5 * 5, out_features=512, key=key3),
            jax.nn.relu,
            eqx.nn.Linear(in_features=512, out_features=64, key=key4),
            jax.nn.relu,
            eqx.nn.Linear(in_features=64, out_features=10, key=key5),
            jax.nn.log_softmax
        ]

    def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "10"]: # forward pass
        for layer in self.layers:
            x=layer(x)
            # print(f"{type(layer).__name__}: {x.shape}")

        return x
