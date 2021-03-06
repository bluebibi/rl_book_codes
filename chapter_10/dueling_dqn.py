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

class DuelingNetwork(nn.Module):
    
    def __init__(self, state_size, action_size, seed):
        
        super(DuelingNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        
        # Input 계층
        self.fc1 = nn.Linear(state_size, 64)
        self.relu1 = nn.ReLU()
        
        # Hindden 계층
        self.fc2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        
        # V(s)
        self.fc3_to_state_value = nn.Linear(64, 1)
        
        # A(s, a)
        self.fc3_to_action_value = nn.Linear(64, self.action_size)
        
    def forward(self, state):
        
        x = self.fc1(state)
        x = self.relu1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        
        v_x = self.fc3_to_state_value(x)
        
        a_x = self.fc3_to_action_value(x)
        
        # average
        average_operator = (1 / self.action_size) * a_x
        
        x = v_x + ( a_x - average_operator ) 
                
        return x

import random
import torch.optim as optim

BUFFER_SIZE = int(100000) # replay buffer size
BATCH_SIZE = 64           # minibatch size
GAMMA = 0.99              # discount factor
TAU = 1e-3                # for soft update of target parameters
LR = 0.0005               # learning rate
UPDATE_EVERY = 4          # how often to update the network

from collections import deque, namedtuple
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PrioritizedReplayBuffer(object):
    """
    https://github.com/higgsfield/RL-Adventure/blob/master/4.prioritized%20dqn.ipynb
    """
    def __init__(self, buffer_size, batch_size, prob_alpha=0.6, prob_beta=0.5):
        self.prob_alpha = prob_alpha
        self.prob_beta = prob_beta
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((buffer_size, ), dtype=np.float32) # long array
        self.experience = namedtuple("Experience", field_names=["state", 
                                                                "action", 
                                                                "reward", 
                                                                "next_state",
                                                                "done"])
        
    def add(self, state, action, reward, next_state, done):
        
        # if self.buffer is empty return 1.0, else max
        max_priority = self.priorities.max() if self.buffer else 1.0
        exp = self.experience(state, action, reward, next_state, done)
        
        # if buffer has rooms left
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(exp)
        else:
            self.buffer[self.pos] = exp
        
        # assign max priority
        self.priorities[self.pos] = max_priority
        
        # update index
        self.pos = (self.pos + 1) % self.buffer_size 
        
    def sample(self, completion):
        
        beta = self.prob_beta + (1-self.prob_beta) * completion
        
        # if buffer is maxed out..
        if len(self.buffer) == self.buffer_size:
            # all priorities are the same as self.priorities
            priorities = self.priorities
        else:
            # all priorities are up to self.pos cuz it's not full yet
            priorities = self.priorities[:self.pos]
            
        # $ P(i) = (p_i^\alpha) / \Sigma_k p_k^\alpha $
        probabilities_a = priorities ** self.prob_alpha
        sum_probabilties_a = probabilities_a.sum()
        P_i = probabilities_a / sum_probabilties_a
        
        sampled_indices = np.random.choice(len(self.buffer), self.batch_size, p=P_i)
        experiences = [self.buffer[idx] for idx in sampled_indices]
        
        # $ w_i = ( 1/N * 1/P(i) ) ** \beta $
        # $ w_i = ( N * P(i) ** (-1 * \beta) ) $
        N = len(self.buffer)
        weights = ( N * P_i[sampled_indices] ) ** (-1 * beta)
        
        #  For stability reasons, we always normalize weights by 1/ maxi wi so
        #  that they only scale the update downwards.
        weights = weights / weights.max()
        
        states = torch.from_numpy(np.vstack([exp.state for exp in experiences if exp is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([exp.action for exp in experiences if exp is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([exp.reward for exp in experiences if exp is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([exp.next_state for exp in experiences if exp is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([exp.done for exp in experiences if exp is not None]).astype(np.uint8)).float().to(device)
        weights = torch.from_numpy(np.vstack(weights)).float()
        
        return states, actions, rewards, next_states, dones, sampled_indices, weights
        
    def update_priorities(self, batch_indicies, batch_priorities):
        for idx, priority in zip(batch_indicies, batch_priorities):
            self.priorities[idx] = priority
        
    def __len__(self):
        return len(self.buffer)


class DNPERAgent():
    
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        # 메인 Q, 고정된 타깃 네트워크 초기화
        self.qnetwork_local = DuelingNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = DuelingNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        
        # 리플레이 메모리
        self.memory = PrioritizedReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        
        self.t_step = 0
    
    
    def step(self, state, action, reward, next_state, done, completion):
        
        # 리플레이 메모리에 경험 저장
        self.memory.add(state, action, reward, next_state, done)
    
        # 일정 주기마다 학습 수행
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(completion)
                self.learn(experiences, GAMMA)
    
    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # 메인 Q 네트워크 Evaluate 모드 활성화
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
        
    
    def learn(self, experiences, gamma, p_eps = 1e-5):
        
        # 리플레이 메모리에서 경험 가져오기
        states, actions, rewards, next_states, dones, sampled_indicies, weights = experiences
        
        # 최적 행동을 메인 Q 네트워크에서 찾기
        best_actions = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, best_actions)
        
        # 현재 상태로 고정된 타깃 Q 값을 계산
        Q_targets = rewards + gamma * Q_targets_next * (1-dones)
        
        # 예측된 Q 값을 계산
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # PER을 위한 TD 오차 계산
        TD_Error = Q_targets - Q_expected
        new_priorities = TD_Error.abs().detach().numpy() + p_eps
        
        loss = (TD_Error.pow(2) * weights).mean()
        
        # minimise the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.memory.update_priorities(sampled_indicies, new_priorities)
        self.optimizer.step()
        
        # target network 업데이트
        self.target_update(self.qnetwork_local, self.qnetwork_target, TAU)
    
    def target_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)
            


from collections import deque

def ddqn_with_per(agent, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, fname="dqn"):
    
    output_path = "outputs/{}".format(fname)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    scores = [] # list containing scores from each episode
    scores_window = deque(maxlen=10) # last 100 scores
    eps = eps_start # initialize epsilon
    save_score_threshold = 200
    
    # for every episode..
    for i_episode in range(1, n_episodes + 1):
        
        # episode completion
        completion = i_episode / n_episodes
        
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
            agent.step(state, action, reward, next_state, done, completion)
            
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
            break

env = gym.make('CartPole-v0')
env.seed(0)
agent = DNPERAgent(state_size=4, action_size=2, seed=0)
ddqn_with_per(agent, fname='dn_per')

