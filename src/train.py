import jax
import jax.numpy as jnp
import equinox as eqx
import optax  # https://github.com/deepmind/optax

import torch.utils.data.dataloader
from jaxtyping import Float, Array, Int, PyTree

from model import CNN
from constants import LEARNING_RATE, TRAINING_STEPS, PRINT_EVERY
from data import get_trainloader, get_testloader
from evaluate import evaluate
from utils import loss

def train(
    model: CNN,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    optim: optax.GradientTransformation,
    steps: int,
    print_every: int,
) -> CNN:
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(
        model: CNN,
        opt_state: PyTree,
        x: Float[Array, "batch 1 28 28"],
        y: Int[Array, " batch"],
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    # Loop over our training dataset as many times as we need.
    def infinite_trainloader():
        while True:
            yield from trainloader

    for step, (x, y) in zip(range(steps), infinite_trainloader()):
        # PyTorch dataloaders give PyTorch tensors by default,
        # so convert them to NumPy arrays.
        x = x.numpy()
        y = y.numpy()
        model, opt_state, train_loss = make_step(model, opt_state, x, y)
        if (step % print_every) == 0 or (step == steps - 1):
            test_loss, test_accuracy = evaluate(model, testloader)
            print(
                f"{step=}, train_loss={train_loss.item()}, "
                f"test_loss={test_loss.item()}, test_accuracy={test_accuracy.item()}"
            )
    return model

if __name__ == "__main__":
    from constants import SEED
    model = CNN(jax.random.PRNGKey(SEED))
    optim = optax.adamw(LEARNING_RATE)
    model = train(model, get_trainloader(), get_testloader(), optim, TRAINING_STEPS, PRINT_EVERY)
