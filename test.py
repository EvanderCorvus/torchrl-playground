import numpy as np
import torch as tr
from agents import SAC
from tensordict import TensorDict
from utils.utils import *
from torchrl.envs import GymEnv
from torchrl.envs import GymEnv, StepCounter, TransformedEnv
from torchrl.collectors import SyncDataCollector
from tqdm import tqdm
from torchrl.record import CSVLogger, VideoRecorder
from torch.utils.tensorboard import SummaryWriter
import os
from tensordict import TensorDict


device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')

config = hyperparams_dict('SAC')
env_config = hyperparams_dict('Environment')
env = TransformedEnv(
    GymEnv("Pendulum-v1"),
    StepCounter(max_steps=env_config['n_steps']),
    device=device
)
agent = SAC(config, env.action_spec)

state = tr.rand(1, 3)
obs = TensorDict({'observation': state},
                 batch_size=1,
                 device=device)
for _ in range(3):
    action = agent.actor(obs)
    print(action)
    break

