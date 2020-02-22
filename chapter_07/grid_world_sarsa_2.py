# 사용 패키지 임포트
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt

from environments.gridworld import GridWorld

plt.rcParams["font.family"] = 'AppleGothic'
plt.rcParams["font.size"] = 12
mpl.rcParams['axes.unicode_minus'] = False

STEP_N_MAX = 9


# 행동-가치 함수 생성
def state_action_value(env):
    q = dict()
    for state in env.STATES:
        for action in env.ACTIONS:
            q[(state, action)] = np.random.normal()
    return q


# 탐욕적 정책을 생성하는 함수
def generate_greedy_policy(env, Q):
    policy = dict()
    for state in env.STATES:
        actions = []
        q_values = []
        prob = []

        for action in env.ACTIONS:
            actions.append(action)
            q_values.append(Q[state, action])

        for i in range(len(q_values)):
            if i == np.argmax(q_values):
                prob.append(1)
            else:
                prob.append(0)

        policy[state] = (actions, prob)
    return policy


# ε-탐욕적 정책의 확률 계산 함수
def e_greedy(env, e, q, state):
    action_values = []
    prob = []
    for action in env.ACTIONS:
        action_values.append(q[(state, action)])

    for i in range(len(action_values)):
        if i == np.argmax(action_values):
            prob.append(1 - e + e/len(action_values))
        else:
            prob.append(e/len(action_values))
    return env.ACTIONS, prob


# ε-탐욕적 정책 생성 함수
def generate_e_greedy_policy(env, e, Q):
    policy = dict()
    for state in env.STATES:
        policy[state] = e_greedy(env, e, Q, state)
    return policy


# n-스텝 SARSA 함수
# 초기 하이퍼파라미터 설정: ε=0.3, α=0.5, γ=0.98, n-스텝 = 3, 반복 수행 횟수 = 100
def n_step_sarsa(env, epsilon=0.3, alpha=0.5, gamma=0.98, n=3, num_iter=100, learn_policy=True):
    Q = state_action_value(env)
    policy = generate_e_greedy_policy(env, epsilon, Q)

    cumulative_reward = 0
    average_reward = []

    for episode in range(num_iter):
        current_state = env.reset()
        action = np.random.choice(policy[current_state][0], p=policy[current_state][1])
        state_trace, action_trace, reward_trace = [current_state], [action], []
        t, T = 0, 10000

        # SARSA == STATE ACTION REWARD STATE ACTION
        while True:
            if t < T:
                reward, next_state = env.step(current_state, action)
                reward_trace.append(reward)
                state_trace.append(next_state)

                if next_state in env.GOAL_STATES:
                    T = t + 1
                    cumulative_reward += sum(reward_trace)      # episode 누적 reward
                else:
                    next_action = np.random.choice(policy[next_state][0], p=policy[next_state][1])
                    action_trace.append(next_action)

            tau = t - n + 1
            if tau >= 0:
                G = 0
                for i in range(tau + 1, min([tau + n, T]) + 1):
                    G += (gamma ** (i - tau - 1)) * reward_trace[i - 1]

                if tau + n < T:
                    G += (gamma ** n) * Q[state_trace[tau + n], action_trace[tau + n]]

                Q[state_trace[tau], action_trace[tau]] += alpha * (G - Q[state_trace[tau], action_trace[tau]])

                if learn_policy:
                    policy[state_trace[tau]] = e_greedy(env, epsilon, Q, state_trace[tau])

            current_state = next_state
            action = next_action

            if tau == (T - 1):
                break
            t += 1

        # max episode를 10등분하여 그 위치에 해당하는 reward 저장
        if (episode+1) % (num_iter/10) == 0:
            average_reward.append(cumulative_reward/(episode+1))

    return policy, Q, average_reward


if __name__ == '__main__':
    # 그리드 월드 환경 객체 생성
    env = GridWorld(transition_reward=-0.1)

    step_n = np.power(2, np.arange(0, 10))
    max_episode = 100
    episodes = np.arange(max_episode/10, max_episode+1, max_episode/10)

    average_rewards = np.zeros((len(step_n), len(episodes)))
    for n_idx, n in enumerate(step_n):
        average_reward_lst = []
        for _ in range(100):
            policy, Q, average_reward = n_step_sarsa(env, epsilon=0.2, alpha=0.2, gamma=0.98, n=n, num_iter=max_episode)
            average_reward_lst.append(average_reward)

        for episode_idx, episode in enumerate(episodes):
            reward_sum = 0
            for i in range(len(average_reward)):
                reward_sum += average_reward_lst[i][episode_idx]
            average_rewards[n_idx, episode_idx] = reward_sum/len(average_reward)
            print("step_n:", n, " episodes:", episode)
            # print(policy)
            # print(Q)
            print("average_reward:", average_rewards[n_idx, episode_idx])

    marker = ['o', 'x', '.', 's', '*', '+', '|', '^', 'D', ' ']
    for i in range(0, len(step_n)):
        plt.plot(episodes, average_rewards[i, :], marker=marker[i], label='n = %d' % (step_n[i]))

    plt.xlabel('진행된 episode')
    plt.ylabel('episode 누적 평균 reward')
    plt.ylim([-7, 0])
    plt.legend()

    plt.savefig('images/n_step_sarsa_for_grid_world_alpha_0,4.png')
    plt.close()
