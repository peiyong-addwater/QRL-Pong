import pufferlib
from pufferlib import vector
from pufferlib.ocean import env_creator
from pufferlib.ocean.pong.pong import Pong


import time

import torch

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    print("CUDA is available. PyTorch can use the GPU.")
    
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    
    # Get the name of the current GPU
    current_gpu_name = torch.cuda.get_device_name(0) # Assuming device 0 is being used
    print(f"Current GPU in use: {current_gpu_name}")
    
    # Create a tensor and move it to the GPU to confirm
    device = torch.device("cuda")
    x = torch.randn(2, 3).to(device)
    print(f"Tensor created on GPU: {x}")
    print(f"Is tensor on CUDA device? {x.is_cuda}")
    
    # Perform a simple operation on the GPU
    y = x * 2
    print(f"Result of operation on GPU: {y}")
    
else:
    print("CUDA is not available. PyTorch will use the CPU for computations.")

print("==========Starting Pong environment...==========")


env_name = 'puffer_pong'
env_creator = env_creator(env_name)
vecenv = vector.make(env_creator, num_envs=2, num_workers=2, batch_size=1,
        backend=pufferlib.vector.Multiprocessing, env_kwargs={'num_envs': 4, 'log_interval': 1, 'frameskip':4})
print("Action space:", vecenv.single_action_space.n)
print("Observation space:", vecenv.single_observation_space.shape[0])

obs, _ = vecenv.reset()

test_count = 0
while True:
    print(f"==========Step {test_count+1} in Pong environment...==========")
    action = [vecenv.single_action_space.sample() for _ in range(vecenv.num_envs)]
    obs, reward, terminated, truncated, info = vecenv.step(action)
    # print("Observation:", obs) # paddly_yl, paddle_yr, ball_x, ball_y, ball_vx, ball_vy, score_l, score_r
    # right paddle is controlled by the agent
    print("Observation shape:", obs.shape)
    print("Action shape:", len(action))
    print("Reward shape:", reward.shape)
    print("Terminated shape:", terminated.shape)
    print("Truncated shape:", truncated.shape)
    #print("Info shape:", len(info))
    #print(vars(vecenv))
    paddle_yl, paddle_yr, ball_x, ball_y, ball_vx, ball_vy, score_l, score_r = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3], obs[:, 4], obs[:, 5], obs[:, 6], obs[:, 7]
    #print("Paddle YR:", paddle_yr, "; Paddle YL:", paddle_yl, "; Ball X:", ball_x, "; Ball Y:", ball_y, "; Ball VX:", ball_vx, "; Ball VY:", ball_vy, "; Score L:", score_l, "; Score R:", score_r)
    print("Score Left:\n", score_l, "\nScore Right:\n", score_r)
    # print("Action taken:", action)
    print("Reward:", reward) # it seems the agent will be rewarded for hitting the ball, even without scoring.
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Info:", info) # something like [{'perf': 0.0, 'score': -21.0, 'episode_return': -21.0, 'episode_length': 1.0, 'n': 1.0}]
    time.sleep(1)
    test_count += 1
    if test_count > 1000:
        break
vecenv.close()