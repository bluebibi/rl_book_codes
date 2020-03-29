# Import the necessary package

import os
import pickle

import gym
import random
import torch
import numpy as np
from collections import deque

import matplotlib.pyplot as plt

# Set up the environment
env = gym.make('CartPole-v0')
env.seed(0)
print('State shape: {}'.format(env.observation_space.shape))
print('Number of actions: {}'.format(env.action_space.n))

# Model define
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
        
    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        #fc1 8 -> 64
        self.fc1 = nn.Linear(state_size, 32)
        
        #relu1
        self.relu1 = nn.PReLU()
        
        #fc2 64 -> 64
        self.fc2 = nn.Linear(32, 32)
        
        #relu2
        self.relu2 = nn.PReLU()
        
        #fc3 64 -> action_size
        self.fc3 = nn.Linear(32, action_size)
        
    def forward(self, state):
        #input state: [bsz, 8]
        
        #fc1 8 -> 64
        x = self.fc1(state)
        
        #relu1
        x = self.relu1(x)
        
        #fc2 64 -> 64
        x = self.fc2(x)
        
        #relu2
        x = self.relu2(x)
        
        #fc3 64 -> action_size
        x = self.fc3(x)
        
        return x


# Define agent
import random
import torch.optim as optim

BUFFER_SIZE = int(100000) # replay buffer size
BATCH_SIZE = 64           # minibatch size
GAMMA = 0.99              # discount factor
LR = 0.0005               # learning rate
UPDATE_EVERY = 4          # how often to update the network

import random
import torch.optim as optim

BUFFER_SIZE = int(100000) # replay buffer siz›e
BATCH_SIZE = 64           # minibatch size
GAMMA = 0.99              # discount factor
LR = 0.0005               # learning rate
UPDATE_EVERY = 4          # how often to update the network

class DDQNAgent():
    
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        # initialize Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        
        # replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
        # initialize time step
        self.t_step = 0
    
    
    def step(self, state, action, reward, next_state, done):
        # save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
    
        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
    
    def act(self, state, eps=0.):
        # single state to state tensor (batch size = 1)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # set eval mode for local QN 
        self.qnetwork_local.eval()
        
        # predict state value with local QN
        with torch.no_grad(): # no need to save the gradient value
            action_values = self.qnetwork_local(state)
        
        # set the mode of local QN back to train
        self.qnetwork_local.train()
        
        # e-greedy action selection
        # return greedy action if prob > eps
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        
        # return random action if prob <= eps
        else:
            return random.choice(np.arange(self.action_size))
        
    
    def learn(self, experiences, gamma):
        
        # unpack epxeriences
        states, actions, rewards, next_states, dones = experiences
        
        # define loss function: MSELoss
        loss_function = torch.nn.MSELoss()
        
        best_actions = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, best_actions)
        
        # compute Q targets from current states
        Q_targets = rewards + gamma * Q_targets_next * (1-dones)
        
        # get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # compute loss
        loss = loss_function(Q_expected, Q_targets)
        
        # minimise the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # update target network
        self.target_update(self.qnetwork_local, self.qnetwork_target)
    
    def target_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

from collections import deque, namedtuple
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    
    # initialize ReplayBuffer
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", 
                                                                "action", 
                                                                "reward", 
                                                                "next_state", 
                                                                "done"])
        self.seed = random.seed(seed)
    
    # add a new experience to memory
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    # randomly sample a batch of experiences from memory
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    # return the size of the internal memory
    def __len__(self):
        return len(self.memory)

# Train agent
from collections import deque

def dqn(agent, n_episodes=2000, max_t=200, eps_start=1.0, eps_end=0.01, eps_decay=0.995, fname="dqn"):
    
    output_path = "outputs/{}".format(fname)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    scores = [] # list containing scores from each episode
    scores_window = deque(maxlen=10) # last 100 scores
    eps = eps_start # initialize epsilon
    save_score_threshold = 200
    
    # for every episode..
    for i_episode in range(1, n_episodes + 1):
        
        # reset state
        state = env.reset()
        
        # reset score to 0
        score = 0
        
        # for every time step until max_t
        for t in range(max_t):
            
            # get action based on e-greedy policy
            action = agent.act(state, eps)
            
            # execute the chosen action
            next_state, reward, done, _ = env.step(action)
            
            # update the network with experience replay
            agent.step(state, action, reward, next_state, done)
            
            # set next_state as the new state
            state = next_state
            
            # add reward to the score
            score += reward
            
            # if the agent has reached the terminal state, break the loop
            if done:
                break
        
        # append the episode score to the deque
        scores_window.append(score)
        
        # append the episode score to the list
        scores.append(score)
        
        # decrease episilon
        eps = max(eps_end, eps_decay * eps)
        
        # display metrics
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        
        # save model if the latest average score is higher than 200.0
        if np.mean(scores_window) >= save_score_threshold:
            print('\nEnvironment solved in {:d} episodes! \tAverage Score: {:.2f}'.format(i_episode-10, np.mean(scores_window)))
            print('Finish the DDQN agent train!')
            break

env = gym.make('CartPole-v0')
env.seed(0)
agent = DDQNAgent(state_size=4, action_size=2, seed=10)
dqn(agent, fname='ddqn')

