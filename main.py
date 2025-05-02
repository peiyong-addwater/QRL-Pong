from pettingzoo.atari import pong_v3
import gymnasium as gym
import supersuit as ss

env = pong_v3.parallel_env(render_mode="human")
#env = ss.frame_skip_v0(env, 4)
env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
env = ss.color_reduction_v0(env, mode="B")
env = ss.resize_v1(env, x_size=84, y_size=84)
env = ss.frame_stack_v1(env, 4)
env = ss.agent_indicator_v0(env, type_only=False)
#env = ss.pettingzoo_env_to_vec_env_v1(env)
#env = ss.concat_vec_envs_v1(env, 16 // 2, num_cpus=0, base_class="gymnasium")
env.single_observation_space = env.observation_space('first_0')
env.single_action_space = env.action_space('first_0')
print(env.single_observation_space)
print(env.single_action_space.n)
#print(env.observation_space('first_0'))
#print(env.action_space('first_0'))
#print(env.observation_space('second_0'))
#print(env.action_space('second_0'))
env.is_vector_env = True
observations, infos = env.reset(seed=0)
print(observations)
print(infos)
#print(env.agents)
#print(env.action_space)
while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    print(actions)
    observations, rewards, terminations, truncations, infos = env.step(actions)
    #print(observations)
    print(rewards)
    print(terminations)
    print(truncations)
    print(infos)
    break
env.close()

