import torch as tr
from tensordict import TensorDict
import torch.nn as nn
from tensordict.nn import TensorDictModule as tdmod, TensorDictSequential as tdseq
from torchrl.modules.distributions.continuous import NormalParamWrapper, TanhNormal
from torchrl.modules.tensordict_module.actors import ProbabilisticActor
from torchrl.modules import ActorCriticOperator, ValueOperator, MLP
from torchrl.modules import SafeModule
from torchrl.data.replay_buffers import LazyMemmapStorage, ReplayBuffer
from torchrl.objectives.sac import SACLoss
from torchrl.objectives import SoftUpdate
from torchrl.modules.tensordict_module.common import SafeModule
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')


class Critic(nn.Module):
    def __init__(self, config):
        super(Critic, self).__init__()
        self.net = MLP(config['obs_dim'] + config['act_dim'],
                       1,
                       depth=config['depth_critic'],
                       num_cells=config['hidden_dim_critic'],
                       activation_class=nn.ReLU,
                       device=device,
                       )

    def forward(self, state, action):
        x = tr.cat([state, action], dim=-1)
        return self.net(x)


def create_ActorCritic(config, action_spec):
    net = NormalParamWrapper(
        MLP(config['obs_dim'], 2 * config['act_dim'],
            depth=config['depth_actor'],
            num_cells=config['hidden_dim_actor'],
            # activation_class=nn.ReLU,
            device=device),
    )
    module_policy = SafeModule(net,
                               in_keys=['observation'],
                               out_keys=['loc', 'scale']
                               )
    td_module_policy = ProbabilisticActor(
        module=module_policy,
        in_keys=['loc', 'scale'],
        spec=action_spec,
        distribution_class=TanhNormal,
        return_log_prob=True,
    )
    module_value = Critic(config)

    td_module_value = ValueOperator(
        module=module_value,
        in_keys=['observation', 'action'],
    )
    return td_module_policy, td_module_value


class SAC:
    def __init__(self, config, action_spec):
        self.batch_size = config['batch_size']
        self.actor, self.critic = create_ActorCritic(config, action_spec)

        self.replay_buffer = ReplayBuffer(
            storage=LazyMemmapStorage(max_size=config['buffer_size']),
        )
        self.criterion = SACLoss(
            actor_network=self.actor,
            qvalue_network=self.critic,
            alpha_init=config['entropy_coeff'],
            target_entropy=-config['act_dim'],
            fixed_alpha=config['fixed_entropy'],
        )
        self.optimizer_actor = tr.optim.Adam(
            self.actor.parameters(),
            lr=config['lr_actor'],
        )
        self.optimizer_critic = tr.optim.Adam(
            self.critic.parameters(),
            lr=config['lr_critic'],
        )
        self.updater = SoftUpdate(
            self.criterion,
            eps=config['polyak_tau'],
        )
        self.scheduler_actor = ReduceLROnPlateau(self.optimizer_actor,
                                                 mode='min',
                                                 factor=config['gamma_actor'],
                                                 patience=10,
                                                 )
        self.scheduler_critic = ReduceLROnPlateau(self.optimizer_critic,
                                                  mode='min',
                                                  factor=config['gamma_critic'],
                                                  patience=10,
                                                  )


    def update(self):
        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()

        sample = self.replay_buffer.sample(self.batch_size).to(device)

        loss_vals = self.criterion(sample)
        loss = loss_vals['loss_actor'] + loss_vals['loss_qvalue']
        loss.backward()

        self.optimizer_critic.step()
        self.optimizer_actor.step()
        self.updater.step()

        return loss_vals['loss_actor'].item(), loss_vals['loss_qvalue'].item()

    def save(self, path, name):
        state_dict = self.criterion.state_dict()
        tr.save(state_dict, f'{path}/{name}.pth')

    def load(self, path, name):
        state_dict = tr.load(f'{path}/{name}.pth')
        self.criterion.load_state_dict(state_dict)


'''
def create_ActorCriticOperator(config):
    module_hidden = MLP(config['obs_dim'], config['state_dim'],
                        depth=config['depth_common'],
                        num_cells=config['hidden_dim_common'],
                        #activation_class=nn.ReLU
                        )
    td_module_hidden = SafeModule(
        module=module_hidden,
        in_keys=['observation'],
        out_keys=['latent_state'],
    )
    module_policy = NormalParamWrapper(
        MLP(config['state_dim'], config['act_dim'],
            depth=config['depth_actor'],
            num_cells=config['hidden_dim_actor']),
    )
    module_policy = tdmod(module_policy, in_keys=['hidden'], out_keys=['loc', 'scale'])
    td_module_policy = ProbabilisticActor(
        module=module_policy,
        in_keys=['loc', 'scale'],
        out_keys=['action'],
        distribution_class=TanhNormal,
        return_log_prob=True,
    )
    module_value = MLP(config['state_dim']+config['act_dim'],
                       1,
                       depth=config['depth_critic'],
                       num_cells=config['hidden_dim_critic'],
                       #activation_class=nn.ReLU
                       )
    td_module_value = ValueOperator(
        module=module_value,
        in_keys=['latent_state', 'action'],
        out_keys=['state_action_value'],
    )
    td_actor_critic = ActorCriticOperator(
        common_operator=td_module_hidden,
        policy_operator=td_module_policy,
        value_operator=td_module_value,
    )
    return td_actor_critic
'''