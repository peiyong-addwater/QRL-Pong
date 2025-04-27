from pettingzoo.atari import pong_v3
import gymnasium as gym

env = pong_v3.parallel_env(render_mode="human")
env = gym.wrappers.ResizeObservation(env, (84, 84))
env = gym.wrappers.GrayscaleObservation(env)
env = gym.wrappers.FrameStackObservation(env, 4)
observations, infos = env.reset()

while env.agents:
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

