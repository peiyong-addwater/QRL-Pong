import numpy as np
import pickle
import gym
from itertools import compress
import sys

# hyperparameters
H = int(sys.argv[1]) 
batch_size =  10
learning_rate = 1e-3
gamma = 0.99 
decay_rate = 0.99 
resume = False
render = False

# model initialization
D = 16 

if resume:
  model = pickle.load(open(sys.argv[2], 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) 
  model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } 
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } 

grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } 

def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x))

def prepro(I,H):
  """ prepro 210x160x3 uint8 frame into 16 1D float vector """

  I = I[35:195] 
  I = I[::2,::2,0] 
  I[I == 144] = 0 
  I[I == 109] = 0 
  I[I != 0] = 1 

  x, y = np.where(I!=0)

  get_ball = (9<y) & (y<70)
  get_p1 = (y==9)
  get_p2 = (y==70)

  bx = list(compress(x,get_ball))
  by = list(compress(y,get_ball))
  Itran = I.transpose()

  if len(bx)!=0:
    by = np.mean(by)
    bx = np.mean(bx)
  else:
    bx, by = 0, 0
  p1f = list(compress(x,get_p1))
  p2f = list(compress(x,get_p2))
  if len(p1f)!=0:
      p1 = np.mean(np.array(p1f))
      p1e1 = np.max(np.array(p1f))
      p1e2 = np.min(np.array(p1f))
  else:
      p1 = 0
      p1e1 = 0
      p1e2 = 0
  if len(p2f)!=0:
      p2 = np.mean(p2f)
      p2e1 = np.max(p2f)
      p2e2 = np.min(p2f)
  else:
      p2 = 0
      p1e1 = 0
      p1e2 = 0

  H = [80*x for x in H]

  ob = [p1e1,p1,p1e2,bx,by,p2e1,p2,p2e2,H[0],H[1],H[2],H[3],H[4],H[5],H[6],H[7]] 
  ob = [x/80 for x in ob]

  return np.ravel(np.array(ob).astype(np.float32))

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0 
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)
  return p, h 

def policy_backward(eph, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  dW2 = np.dot(eph.T, epdlogp).ravel()
  dh = np.outer(epdlogp, model['W2'])
  dh[eph <= 0] = 0 
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}

env = gym.make("PongDeterministic-v4")
observation = env.reset()[0]
prev_x = None 
xs,hs, dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
updating = -1

prev_x = np.zeros(D)
reward_sums = list()
with open(f"polgrad_SS16_ND_BS10_10N{H}_LR3.log", "w") as file:
  while True:
    if render: env.render()
  
    try:
      cur_x = prepro(observation,prev_x)

    except:
      prev_x = np.ravel(np.zeros(D).astype(np.float32))
      cur_x = prepro(observation,prev_x)

    x = cur_x
    prev_x = cur_x
    
    aprob, h = policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3 
  
    xs.append(x) 
    hs.append(h) 
    y = 1 if action == 2 else 0 
    dlogps.append(y - aprob) 
  
    # step the environment and get new measurements
    observation, reward, done, idk, info = env.step(action)
    reward_sum += reward
  
    drs.append(reward) 
  
    if done: 
      episode_number += 1
  
      epx = np.vstack(xs)
      eph = np.vstack(hs)
      epdlogp = np.vstack(dlogps)
      epr = np.vstack(drs)
      xs,hs,dlogps,drs = [],[],[],[] 
  
      discounted_epr = discount_rewards(epr)
      discounted_epr -= np.mean(discounted_epr)
      discounted_epr /= np.std(discounted_epr)
  
      epdlogp *= discounted_epr 
      grad = policy_backward(eph, epdlogp)
      for k in model: grad_buffer[k] += grad[k] 

      if episode_number % batch_size == 0:
        updating +=1
        if updating % 1000 == 0:
          updating = 0
        for k,v in model.items():
          g = grad_buffer[k] # gradient
          rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
          model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
          grad_buffer[k] = np.zeros_like(v)
  
      reward_sums.append(reward_sum)
      avg_reward_sum = sum(reward_sums[-50:]) / len(reward_sums[-50:])
      running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
      file.write(f'{reward_sum},{running_reward} \n')
      print(f'{reward_sum},{avg_reward_sum},{running_reward}')
      file.flush()
      if episode_number % 10 == 0:
        pickle.dump(model, open(f'save_SS16_ND_BS_N{H}_LR3.p', 'wb'))
      reward_sum = 0
      observation = env.reset()[0] # reset env
      prev_x = None
  
