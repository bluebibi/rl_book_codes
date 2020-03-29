import torch
import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Constants
GAMMA = 0.99


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=256):
        super(PolicyNetwork, self).__init__()
        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x), dim=1)
        return x


class REINFORCE:
    def __init__(self, env, learning_rate=3e-4):
        self.env = env
        self.num_actions = self.env.action_space.n
        self.policy_network = PolicyNetwork(self.env.observation_space.shape[0], self.env.action_space.n)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        self.episodes = []
        self.episodes_reward = []

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy_network.forward(Variable(state))
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob

    def update_policy(self, rewards, log_probs):
        discounted_rewards = []

        for t in range(len(rewards)):
            Gt = 0
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + GAMMA ** pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)

        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                    discounted_rewards.std() + 1e-9)  # normalize discounted rewards

        policy_gradient = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)

        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.optimizer.step()

    def train(self, max_episode=3000, max_step=200):
        for episode in range(max_episode):
            state = env.reset()
            log_probs = []
            rewards = []
            episode_reward = 0
            self.episodes.append(episode)

            for steps in range(max_step):
                action, log_prob = self.get_action(state)
                new_state, reward, done, _ = self.env.step(action)

                log_probs.append(log_prob)
                rewards.append(reward)
                episode_reward += reward

                if done:
                    self.update_policy(rewards, log_probs)
                    if episode % 10 == 0:
                        print("episode " + str(episode) + ", reward : " + str(episode_reward))
                    self.episodes_reward.append(episode_reward)

                    break

                state = new_state

    def draw_graph(self):
        plt.rcParams.update({'font.size': 20})
        plt.figure(figsize=(16, 12))
        plt.plot(self.episodes, self.episodes_reward)
        plt.xlabel('Episode')
        plt.ylabel('Total reward on episode')
        plt.grid()
        plt.savefig("./ActorCritic_on_CartPole.png")
        plt.show()


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    REINFORCE = REINFORCE(env)
    REINFORCE.train(3000, 200)
    REINFORCE.draw_graph()
