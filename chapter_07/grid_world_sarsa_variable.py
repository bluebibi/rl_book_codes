# 사용 패키지 임포트
import numpy as np
import gym
import gym_gridworlds
import random
import matplotlib.pyplot as plt

STEP_N_MAX = 9


# 행동-가치 함수 생성
def state_action_value(env):
    q = dict()
    for state in range(1, env.observation_space.n):
        for action in range(env.action_space.n):
            q[(state, action)] = np.random.normal()
    return q


def n_state_action_value(env):
    N = dict()
    for state in range(1, env.observation_space.n):
        for action in range(env.action_space.n):
            for n in range(1, STEP_N_MAX+1):
                N[(state, action, n)] = np.random.normal()
    return N


# 탐욕적 정책을 생성하는 함수
def generate_greedy_policy(env, Q):
    policy = dict()
    for state in range(1, env.observation_space.n):
        actions = []
        q_values = []
        prob = []

        for a in range(env.action_space.n):
            actions.append(a)
            q_values.append(Q[state, a])
        for i in range(len(q_values)):
            if i == np.argmax(q_values):
                prob.append(1)
            else:
                prob.append(0)

        policy[state] = (actions, prob)
    return policy


# ε-탐욕적 정책의 확률 계산 함수
def e_greedy(env, e, q, state):
    actions = [act for act in range(env.action_space.n)]
    action_values = []
    prob = []
    for action in actions:
        action_values.append(q[(state, action)])
    for i in range(len(action_values)):
        if i == np.argmax(action_values):
            prob.append(1 - e + e/len(action_values))
        else:
            prob.append(e/len(action_values))
    return actions, prob


# ε-탐욕적 정책 생성 함수
def generate_e_greedy_policy(env, e, Q):
    policy = dict()
    for state in range(1, env.observation_space.n):
        policy[state] = e_greedy(env, e, Q, state)
    return policy


def step_n_e_greedy(env, e, N, state, action):
    action_values = []
    prob = []
    for n in range(1, STEP_N_MAX+1):
        action_values.append(N[(state, action, n)])
    for action_value in action_values:
        if action_value == max(action_values):
            prob.append((1 - e))
        else:
            prob.append(e/(len(action_values) - 1))
    return prob


def generate_step_n_e_greedy_policy(env, e, N):
    step_n_policy = dict()
    for state in range(1, env.observation_space.n):
        for action in range(env.action_space.n):
            step_n_policy[(state, action)] = step_n_e_greedy(env, e, N, state, action)
    return step_n_policy


# n-스텝 SARSA 함수
# 초기 하이퍼파라미터 설정: ε=0.3, α=0.5, γ=0.98, n-스텝 = 3, 반복 수행 횟수 = 100
def variable_n_step_sarsa(env, epsilon=0.3, alpha=0.5, gamma=0.98, num_iter=100, learn_policy=True):
    Q = state_action_value(env)
    policy = generate_e_greedy_policy(env, epsilon, Q)
    N = n_state_action_value(env)
    step_n_policy = generate_step_n_e_greedy_policy(env, epsilon, N)
    cumulative_reward = 0
    for _ in range(num_iter):
        current_state = random.randrange(1, env.observation_space.n)
        action = np.random.choice(policy[current_state][0], p=policy[current_state][1])
        step_n = np.random.choice([n for n in range(1, STEP_N_MAX+1)], p=step_n_policy[(current_state, action)])
        state_trace, action_trace, reward_trace, step_n_trace = [current_state], [action], [], [step_n]
        t, T = 0, 10000
        update_state = 0
        while True:
            if t < T:
                next_state = int(np.where(env.P[action, current_state] == 1)[0][0])
                reward = env.R[action, current_state]
                state_trace.append(next_state)
                reward_trace.append(reward)
                if next_state == 0:
                    T = t + 1
                    cumulative_reward = sum(reward_trace)
                else:
                    action = np.random.choice(policy[next_state][0], p=policy[next_state][1])
                    action_trace.append(action)
                    step_n = np.random.choice([n for n in range(1, STEP_N_MAX+1)], p=step_n_policy[(current_state, action)])
                    step_n_trace.append(step_n)

            while True:
                n = step_n_trace[update_state]
                tau = t - n + 1
                if tau >= 0:
                    G = 0
                    for i in range(tau + 1, min([tau + n, T]) + 1):
                        G += (gamma ** (i - tau - 1)) * reward_trace[i - 1]

                    if tau + n < T:
                        G += (gamma ** n) * Q[state_trace[tau + n], action_trace[tau + n]]

                    Q[state_trace[tau], action_trace[tau]] += alpha * (G - Q[state_trace[tau], action_trace[tau]])
                    N[state_trace[tau], action_trace[tau], n] += alpha * (G - N[state_trace[tau], action_trace[tau], n])

                    if learn_policy:
                        policy[state_trace[tau]] = e_greedy(env, epsilon, Q, state_trace[tau])
                        step_n_policy[(state_trace[tau], action_trace[tau])] = step_n_e_greedy(env, epsilon, N, state_trace[tau], action_trace[tau])

                    update_state += 1
                else:
                    break

            current_state = next_state

            if tau == (T - 1):
                break
            t += 1

    return policy, Q, cumulative_reward


if __name__ == '__main__':
    cumulative_reward_lst = []

    # 그리드 월드 환경 객체 생성
    env = gym.make('Gridworld-v0')
    for _ in range(1):
        policy, Q, cumulative_reward = variable_n_step_sarsa(env, epsilon=0.2, alpha=0.5, gamma=0.98, num_iter=100)
        cumulative_reward_lst.append(cumulative_reward)
    print(policy)
    print(Q)
    print("average_reward:", sum(cumulative_reward_lst)/len(cumulative_reward_lst))
