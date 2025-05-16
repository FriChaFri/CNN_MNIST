import jax
import optax


from model import CNN, CNN2
from train import train
from data import load_data
from constants import SEED, LEARNING_RATE, TRAINING_STEPS, PRINT_EVERY, MODEL_SAVE_PATH
from utils import save_model


model = CNN(jax.random.PRNGKey(SEED))
optim = optax.adamw(LEARNING_RATE)

trainloader, testloader = load_data()

model = train(model, trainloader, testloader, optim, TRAINING_STEPS, PRINT_EVERY)
save_model(model, MODEL_SAVE_PATH)




model = CNN2(jax.random.PRNGKey(SEED))
optim = optax.adamw(LEARNING_RATE)

trainloader, testloader = load_data()

model = train(model, trainloader, testloader, optim, TRAINING_STEPS, PRINT_EVERY)
save_model(model, 'models/cnn2_model.eqx')


