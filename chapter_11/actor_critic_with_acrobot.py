# 사용 패키지 임포트
import torch
import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Constants, 감가율 적용
GAMMA = 0.99

# 액터-크리틱 신경망 생성 클래스
class ActorCriticNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=256):
        super(ActorCriticNetwork, self).__init__()
        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)

    def forward(self, state_tensor):
        value = F.relu(self.critic_linear1(state_tensor))
        value = self.critic_linear2(value)

        policy_dist = F.relu(self.actor_linear1(state_tensor))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

        return value, policy_dist

# 액터 크리틱 클래스
class ActorCritic():
    def __init__(self, env, learning_rate=3e-4):
        self.env = env
        self.num_actions = self.env.action_space.n
        self.ac_net = ActorCriticNetwork(self.env.observation_space.shape[0], self.env.action_space.n)
        self.ac_optimizer = optim.Adam(self.ac_net.parameters(), lr=learning_rate)

        self.episodes = []
        self.episodes_reward = []

    # 액터-크리틱 신경망 파라미터 업데이트 함수
    def update(self, rewards, values, next_value, log_probs, masks, n_steps):
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

    # 액터-크리틱 신경망 출력 함수
    def get_ac_output(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        value, policy_dist = self.ac_net.forward(state)
        action = np.random.choice(self.num_actions, p=policy_dist.detach().numpy().squeeze(0))

        return action, policy_dist, value

    # 학습 함수
    def train(self, batch_size, n_steps, max_episode):
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
            done = False
            while not done:
                k += 1
                action, policy_dist, value = self.get_ac_output(state)
                new_state, reward, done, _ = self.env.step(action)

                log_prob = torch.log(policy_dist.squeeze(0)[action])

                rewards.append(reward)
                values.append(value.detach().numpy()[0])
                log_probs.append(log_prob)
                masks.append(1 - done)
                state = new_state
                episode_reward += reward

                # 에피소드의 스텝 길이가 배치크기가 될 때마다
                # 혹은 에피소드 종료시 업데이트 수행
                if k % batch_size == 0 or done:
                    k = 0
                    _, _, next_value = self.get_ac_output(state)
                    self.update(rewards, values, next_value, log_probs, masks, n_steps)

                    rewards = []
                    values = []
                    log_probs = []
                    masks = []

                if done:
                    self.episodes_reward.append(episode_reward)
                    if episode % 10 == 0:
                        print("episode: " + str(episode) + ", reward : " + str(episode_reward))
                    break

    # 학습 결과 그래프 출력 함수
    def draw_graph(self):
        plt.rcParams.update({'font.size': 20})
        plt.figure(figsize=(16, 12))
        plt.plot(self.episodes, self.episodes_reward)
        plt.xlabel('Episode')
        plt.ylabel('Total reward on episode')
        plt.grid()
        plt.savefig("./Actor-Critic_on_Acrobot.png")
        plt.show()

# main 함수
if __name__ == '__main__':
    # Acrobot 환경 객체 생성
    env = gym.make("Acrobot-v1")

    ac = ActorCritic(env)
    # 액터-크리틱 학습 수행(배치크기=32, $n$-스텝=16, 최대 에피소드=50000)
    ac.train(batch_size=32, n_steps=16, max_episode=3000)
    ac.draw_graph()