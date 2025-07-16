from pufferlib.ocean.pong import pong

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


env = pong.Pong()
obs, _ = env.reset()
test_count = 0
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
    test_count += 1
    if test_count > 10:
        break
env.close()