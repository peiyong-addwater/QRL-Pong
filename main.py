from pettingzoo.atari import pong_v3
import gymnasium as gym
import supersuit as ss

env = pong_v3.parallel_env(render_mode="human",num_players=2)
print(vars(env))
env = ss.frame_skip_v0(env, 4)
env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
env = ss.color_reduction_v0(env, mode="B")
env = ss.resize_v1(env, x_size=84, y_size=84)
env = ss.frame_stack_v1(env, 4)
env = ss.agent_indicator_v0(env, type_only=False)
env = ss.pettingzoo_env_to_vec_env_v1(env)
envs = ss.concat_vec_envs_v1(env, 16 // 2, num_cpus=0, base_class="gymnasium")
envs.single_observation_space = envs.observation_space
# print(envs.single_observation_space)
envs.single_action_space = envs.action_space
envs.is_vector_env = True
observations, infos = envs.reset()
print(vars(envs))
print(vars(envs.vec_envs[0]))
print(vars(envs.vec_envs[0].par_env))
while envs.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    print(actions)
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(observations)
    print(rewards)
    print(terminations)
    print(truncations)
    print(infos)
    break
env.close()

