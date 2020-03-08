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


class ActorCriticNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size1=64, hidden_size2=256):
        super(ActorCriticNetwork, self).__init__()
        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size1)
        self.critic_linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.critic_linear3 = nn.Linear(hidden_size2, 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size1)
        self.actor_linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.actor_mu = nn.Linear(hidden_size2, num_actions)
        self.actor_sigma = nn.Linear(hidden_size2, num_actions)

    def forward(self, state_tensor):
        value = F.relu(self.critic_linear1(state_tensor))
        value = F.relu(self.critic_linear2(value))
        value = self.critic_linear3(value)

        policy_dist = F.relu(self.actor_linear1(state_tensor))
        policy_dist = F.relu(self.actor_linear2(policy_dist))
        mu = 2 * F.tanh(self.actor_mu(policy_dist))
        sigma = F.softplus(self.actor_sigma(policy_dist)) + 1e-3

        return value, mu, sigma


class ActorCritic():
    def __init__(self, env, learning_rate=1e-4):
        self.env = env
        self.num_actions = self.env.action_space.shape[0]
        self.ac_net = ActorCriticNetwork(self.env.observation_space.shape[0], self.env.action_space.shape[0])
        self.ac_net = self.ac_net
        self.ac_optimizer = optim.Adam(self.ac_net.parameters(), lr=learning_rate)
        self.distribution = torch.distributions.Normal

        self.episodes = []
        self.episodes_reward = []

    def update(self, rewards, values, next_value, log_probs, masks, n_steps):
        rewards = torch.tensor(rewards)
        rewards = torch.tanh(rewards / 16)

        qvals = np.zeros(len(values))
        qval = next_value
        for t in reversed(range(len(rewards))):
            if len(rewards) >= (t + n_steps):
                tau = 3
            else:
                tau = len(rewards) - t
            for i in range(tau):
                qval = rewards[t + i] + (GAMMA ** (i + 1)) * qval * masks[t + i]
                if masks[t + i] == 0.0:
                    break
            qvals[t] = qval

        values = torch.FloatTensor(values).squeeze(1)
        qvals = torch.FloatTensor(qvals)
        log_probs = torch.stack(log_probs)

        advantage = qvals - values
        actor_loss = (-log_probs * advantage.detach()).mean()
        critic_loss = F.smooth_l1_loss(values, qvals.detach())
        ac_loss = actor_loss + 0.5 * critic_loss

        self.ac_optimizer.zero_grad()
        ac_loss.backward()
        self.ac_optimizer.step()

        return ac_loss.item()

    def get_ac_output(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        value, mu, sigma = self.ac_net.forward(state)

        normal_dist = self.distribution(mu, sigma)
        action = normal_dist.sample()
        log_prob = normal_dist.log_prob(action)
        action = torch.clamp(action, min=-2., max=2.)
        return action, log_prob, value

    def train(self, batch_size, n_steps, max_episode, max_step):
        batch_size = batch_size + n_steps
        for episode in range(max_episode):
            rewards = []
            values = []
            log_probs = []
            masks = []
            episode_reward = 0
            self.episodes.append(episode)

            state = self.env.reset()
            k = 0
            for steps in range(max_step):
                k += 1
                action, log_prob, value = self.get_ac_output(state)
                new_state, reward, done, _ = self.env.step(action)

                rewards.append(reward)
                values.append(value.detach().numpy()[0])
                log_probs.append(log_prob)
                masks.append(1 - done)
                state = new_state
                episode_reward += reward

                if k % batch_size == 0 or steps == max_step - 1:
                    k = 0
                    _, _, next_value = self.get_ac_output(state)
                    ac_loss = self.update(rewards, values, next_value, log_probs, masks, n_steps)

                    rewards = []
                    values = []
                    log_probs = []
                    masks = []

                if done:
                    self.episodes_reward.append(episode_reward)
                    if episode % 10 == 0:
                        print("episode: " + str(episode) + ", reward : " + str(episode_reward.item()) + ", action: " + str(action.item()) + ", loss: " + str(ac_loss))
                    if episode != 0 and episode % 2000 == 0:
                        self.draw_graph(episode)
                    break

    def draw_graph(self, episode):
        plt.rcParams.update({'font.size': 20})
        plt.figure(figsize=(16, 12))
        N = 10
        sma_reward = np.convolve(self.episodes_reward, np.ones((N,)) / N, mode='valid')
        plt.plot(self.episodes[len(self.episodes) - len(sma_reward):], sma_reward)
        plt.xlabel('Episode')
        plt.ylabel('SMA (N=10) reward on episode')
        plt.grid()
        plt.savefig("./Actor-Critic_on_Pendulum_{}.png".format(episode))
        plt.show()


if __name__ == '__main__':
    env = gym.make("Pendulum-v0")
    ac = ActorCritic(env)
    ac.train(32, 16, 50000, 200)
    ac.draw_graph()
