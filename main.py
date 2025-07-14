from pufferlib.ocean.pong import pong

import time

env = pong.Pong()
obs, _ = env.reset()
while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print("Observation:", obs) # paddly_yl, paddle_yr, ball_x, ball_y, ball_vx, ball_vy, score_l, score_r
    # right paddle is controlled by the agent
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Info:", info)
    time.sleep(1)
