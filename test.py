from agents import create_ActorCritic
from utils.utils import hyperparams_dict
import torch as tr

config = hyperparams_dict('SAC')
actor, critic = create_ActorCritic(config)
print(actor)
print(critic)

