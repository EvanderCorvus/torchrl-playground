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


device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
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
    for i, data in tqdm(enumerate(collector)):
        agent.replay_buffer.extend(data)
        if len(agent.replay_buffer) < config['batch_size']: continue

        for _ in range(config['n_updates']-1):
            loss_vals = agent.update()

        loss_vals = agent.update()

        writer.add_scalar("loss/actor",
                          loss_vals["loss_actor"].item(), i)
        writer.add_scalar("loss/critic",
                          loss_vals["loss_qvalue"].item(), i)
        writer.add_scalar("mean_batch_reward",
                          tr.mean(data['next', 'reward']).item(), i)
        #writer.add_scalar('action',
                          #data['action'][-1].item(), i)
    writer.close()

    with tr.no_grad():
        record_env.rollout(max_steps=800, policy=agent.actor)
    video_recorder.dump()

