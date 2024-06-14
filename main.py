import numpy as np
import torch as tr
from agents import SAC
from tensordict import TensorDict
from utils.utils import *
from utils.train_utils import train_loop
from torchrl.envs import GymEnv
from torchrl.envs import GymEnv, StepCounter, TransformedEnv
from torchrl.collectors import SyncDataCollector
from tqdm import tqdm
from torchrl.record import CSVLogger, VideoRecorder
from torch.utils.tensorboard import SummaryWriter
import os
from dotenv import load_dotenv

load_dotenv('utils/.env')
device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
seed = os.getenv('seed')

np.random.seed(int(seed))
tr.manual_seed(int(seed))

config = hyperparams_dict('SAC')
env_config = hyperparams_dict('Environment')
env = TransformedEnv(
    GymEnv("Pendulum-v1"),
    StepCounter(max_steps=env_config['n_steps']),
    device=device
)

agent = SAC(config, env.action_spec)
collector = SyncDataCollector(
    env,
    policy=agent.actor,
    frames_per_batch=env_config['frames_per_batch'],
    total_frames=env_config['n_total'],
    device=device,
)

train_path = "./train_loop"
exp_number = len(os.listdir(train_path))+1
writer = SummaryWriter(f'{train_path}/{exp_number}')
for key, value in config.items():
    writer.add_text(key, str(value))

test_path = "./test_loop"
logger = CSVLogger(exp_name="sac-pendulum", log_dir=test_path, video_format="mp4")
video_recorder = VideoRecorder(logger, tag="video")
record_env = TransformedEnv(
    GymEnv("Pendulum-v1", from_pixels=True, pixels_only=False, device=device),
    video_recorder,
)

if __name__ == "__main__":

    train_loop(collector, agent, config, writer)
    writer.close()

    with tr.no_grad():
        record_env.rollout(max_steps=800, policy=agent.actor)
    video_recorder.dump()

