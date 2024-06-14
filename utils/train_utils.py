import numpy as np
import torch as tr
from tqdm import tqdm
from ray import train, tune

def train_loop(collector, agent, config, writer=None, scan=False):
    ep_reward = 0
    for i, data in enumerate(collector):
        agent.replay_buffer.extend(data)
        if len(agent.replay_buffer) < config['batch_size']: continue

        for _ in range(config['n_updates']-1):
            agent.update()

        loss_actor, loss_critic = agent.update()

        reward = tr.mean(data['next', 'reward']).item()
        ep_reward += reward
        if writer is not None:
            writer.add_scalar("loss/actor",
                              loss_actor, i)
            writer.add_scalar("loss/critic",
                              loss_critic, i)
            writer.add_scalar("mean_batch_reward",
                              reward, i)
            writer.add_scalar("lr/actor",
                              agent.scheduler_actor.get_last_lr()[0], i)
            writer.add_scalar("lr/critic",
                              agent.scheduler_critic.get_last_lr()[0], i)

        if data['next', 'terminated'].any() or data['next', 'truncated'].any():
            agent.scheduler_actor.step(loss_actor)
            agent.scheduler_critic.step(loss_critic)
            if scan:
                train.report({"loss": loss_actor + loss_critic,
                              "reward": ep_reward})
                ep_reward = 0

    return ep_reward
