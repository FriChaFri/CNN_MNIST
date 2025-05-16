import jax
import jax.numpy as jnp
import equinox as eqx
import optax  # https://github.com/deepmind/optax
import os

import torch.utils.data.dataloader
from jaxtyping import Float, Array, Int, PyTree

from model import CNN
from constants import LEARNING_RATE, TRAINING_STEPS, PRINT_EVERY, MODEL_SAVE_PATH
from data import get_trainloader, get_testloader
from evaluate import evaluate
from utils import loss




"""
🧠 Visual Analogy: What's Happening?
Imagine training is like steering a boat using waves:

The model is the boat.

The optimizer state is your memory of past wave directions and wind speed (used to smooth out the motion).

Each step is a wave hitting the boat (new batch).

grads tell you how much the wave is pushing you off-course.

optax.update(...) says: "Based on how the last few waves looked, here's how we should steer next."

apply_updates() steers the boat by updating the model.

"""

def train(
    model: CNN,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    optim: optax.GradientTransformation,
    steps: int,
    print_every: int,
) -> CNN:
    """
    Takes a model and two dataloaders (train/test)
    runs training for steps number of batches
    applies gradient updates using an optimizer (Optax)
    Prints evalueation every few steps
    returns the trained model
    """
    #AdamW stores momentum and second moment (variance) estimates for each weight
    # So optax needs to know how many parameters there are, what shape each one is, and initialize inter tracking buffers accordingly
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(
        model: CNN,
        opt_state: PyTree,
        x: Float[Array, "batch 1 28 28"],
        y: Int[Array, " batch"],
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
        # optim.update retuns the updates that we need to apply onto the model
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    # Loop over our training dataset as many times as we need.
    def infinite_trainloader():
        while True:
            yield from trainloader

    # this loop gets 1 batch from the trainloader each loop. 
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
    save_model(model, MODEL_SAVE_PATH)