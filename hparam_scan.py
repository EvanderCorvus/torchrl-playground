from agents import SAC
from utils.utils import *
from ray import train, tune
import json
from ray.tune.search.optuna import OptunaSearch
from utils.train_utils import train_loop
#from ray.tune.schedulers import ASHAScheduler
from torchrl.collectors import SyncDataCollector
import os
import ray
import torch as tr
from torchrl.envs import GymEnv, StepCounter, TransformedEnv
from ray.tune.schedulers import ASHAScheduler

temp_path = os.environ['temp_path']
if not tr.cuda.is_available(): raise Exception('CUDA not available.')

def custom_path_name_creator(trial):
    path = f'./{trial.trial_id}'
    return path

env_config = hyperparams_dict('Environment')
def objective(config):
    try:
        device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
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
        reward = train_loop(collector, agent, config, scan=True)

        return reward

    except Exception as e: raise e


config = hyperparams_dict("SAC")
scan_config = hyperparams_dict('hparam-scan')

search_space = {
    'lr_actor': tune.loguniform(1e-6, 1e-3),
    'lr_critic': tune.loguniform(1e-5, 1e-3),
    'entropy_coeff': tune.uniform(1e-5, 3e-2),
}
# Overwrite the default config with the search space
for key in search_space.keys():
    config[key] = search_space[key]

n_episodes = int(env_config['n_total']/env_config['n_steps'])

scheduler = ASHAScheduler(
    max_t=n_episodes,  # The maximum number of training iterations (e.g., epochs)
    grace_period=int(n_episodes//10),    # The number of epochs to run before a trial can be stopped
    reduction_factor=4,  # Reduce the number of trials that factor
)

# Create an Optuna pruner instance
algo = OptunaSearch()

tuner = tune.Tuner(
    tune.with_resources(objective,
                        resources={'cpu': 8, 'gpu': 1}),
    tune_config=tune.TuneConfig(
        metric='reward',
        mode='max',
        search_alg=algo,
        num_samples=scan_config['num_samples'],
        trial_dirname_creator=custom_path_name_creator,
        scheduler=scheduler
    ),
    run_config=train.RunConfig(
        verbose=1,
        failure_config=train.FailureConfig(fail_fast=True),
    ),
    param_space=config
)

if __name__ == "__main__":
    ray.init()
    results = tuner.fit()
    fname = './best_hyperparams.json'
    with open(fname, 'w') as f:
        json.dump(results.get_best_result().config, f)
    print(results.get_best_result().config)
