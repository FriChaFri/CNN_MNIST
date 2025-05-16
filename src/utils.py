from jaxtyping import Float, Int, Array
from model import CNN
import jax
import jax.numpy as jnp

def loss(
    model: CNN, 
    x: Float[Array, "batch 1 28 28"], 
    y: Int[Array, " batch"]
) -> Float[Array, ""]:
    """
    The CNN model is built to take in one image at a time of shape (1,28,28) but our data x is of shape (batch_size, 1, 28, 28)
    jax.vmap fixes this issue. 
    so pred_y is of shape (batch_size, 10) where each row is log-probabilities for digits 0-9
    
    This function just does a forward pass on a batch, then pumps the correct and predicted y values to cross_entropy function
    """
    pred_y = jax.vmap(model)(x)
    return cross_entropy(y, pred_y)

def cross_entropy(
    y: Int[Array, " batch"], 
    pred_y: Float[Array, "batch 10"]
) -> Float[Array, ""]:
    """
    pred_y contains log softmax scores for all cases in batch which is of shape (batch_size, 10)
    y is a vector of true labels, shape is (batch_size,)

    jnp.expand_dims(y,1) makes y of shape (batch_size, 1) so that we can index it properly
    jnp.take_along_axis(pred_y, ..., axis=1) selects the log-probability in pred_y of the correct class from y for each image.
    -jnp.mean() is the average negative log-likelihood across the batch 
    """
    pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y,1),axis=1)
    return -jnp.mean(pred_y)
