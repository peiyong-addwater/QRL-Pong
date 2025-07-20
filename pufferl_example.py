import torch
import pufferlib.vector
import pufferlib.ocean
from pufferlib import pufferl

from PongAgents import (
    PongClassicalCritic,
    GHZAgent,
    SeparableAgent,
    GraphStateAgent
)

from PongAgents.models import AGENTS


# Equivalent to running puffer train puffer_breakout
def cli():
    pufferl.train('puffer_pong')

class Policy(torch.nn.Module):
    def __init__(self, env, agent_type, agent_args: dict = None):
        super().__init__()
        assert agent_type in AGENTS, f"Unknown agent type: {agent_type}"
        self.single_action_dim = env.single_action_space.n
        self.observation_dim = env.single_observation_space.shape[0]
        self.actor = AGENTS[agent_type](
            action_space=self.single_action_dim,
            observation_space=self.observation_dim,
            **agent_args
        )
        self.critic = PongClassicalCritic(input_dim=self.observation_dim)

    def forward_eval(self, observations, state=None):
        logits = self.actor(observations)
        values = self.critic(observations)
        return logits, values

    # We use this to work around a major torch perf issue
    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)

# Managing your own trainer
if __name__ == '__main__':
    import pprint

    env_name = 'puffer_pong'
    agent_type = "graph_state"
    agent_args = {
        "n_layers": 6,
        "post_select": False,
        "edge_list": [(3, 0), (2, 0), (6, 0), (4, 0), (5, 0), (3, 1), (2, 1), (4, 1), (5, 1), (7, 1), (0, 1)]
    }

    env_creator = pufferlib.ocean.env_creator(env_name)
    vecenv = pufferlib.vector.make(env_creator, num_envs=2, num_workers=2, batch_size=1,
        backend=pufferlib.vector.Multiprocessing, env_kwargs={'num_envs': 128})
    policy = Policy(vecenv.driver_env, agent_type=agent_type, agent_args=agent_args).cuda()
    args = pufferl.load_config('default')
    args['train']['env'] = env_name
    args['wandb'] = True
    pprint.pprint(args)

    trainer = pufferl.PuffeRL(args['train'], vecenv, policy)

    for epoch in range(10):
        trainer.evaluate()
        logs = trainer.train()

    trainer.print_dashboard()
    trainer.close()