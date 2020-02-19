# 사용 패키지 임포트
import numpy as np
import random
import matplotlib.pyplot as plt

from environments.gridworld import GridWorld

STEP_N_MAX = 9


# 행동-가치 함수 생성
def state_action_value(env):
    q = dict()
    for state in env.STATES:
        for action in env.ACTIONS:
            q[(state, action)] = np.random.normal()
    return q


def n_state_action_value(env):
    N = dict()
    for state in env.STATES:
        for action in env.ACTIONS:
            for n in range(1, STEP_N_MAX + 1):
                N[(state, action, n)] = np.random.normal()
    return N


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


def step_n_e_greedy(e, N, state, action):
    step_n_values = []
    prob = []
    for n in range(1, STEP_N_MAX + 1):
        step_n_values.append(N[(state, action, n)])

    for action_value in step_n_values:
        if action_value == max(step_n_values):
            prob.append((1 - e + e/len(step_n_values)))
        else:
            prob.append(e/len(step_n_values))
    return prob


def generate_step_n_e_greedy_policy(env, e, N):
    step_n_policy = dict()
    for state in env.STATES:
        for action in env.ACTIONS:
            step_n_policy[(state, action)] = step_n_e_greedy(e, N, state, action)
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
        current_state = env.reset()
        action = np.random.choice(policy[current_state][0], p=policy[current_state][1])
        step_n = np.random.choice(
            [n for n in range(1, STEP_N_MAX + 1)],
            p=step_n_policy[(current_state, action)]
        )
        state_trace, action_trace, reward_trace, step_n_trace = [current_state], [action], [], [step_n]
        t, T = 0, 10000

        update_state = 0

        # SARSA == STATE ACTION REWARD STATE ACTION
        while True:
            if t < T:
                reward, next_state = env.step(current_state, action)
                reward_trace.append(reward)
                state_trace.append(next_state)

                if next_state in env.GOAL_STATES:
                    T = t + 1
                    cumulative_reward += sum(reward_trace)
                else:
                    next_action = np.random.choice(policy[next_state][0], p=policy[next_state][1])
                    next_step_n = np.random.choice(
                        [n for n in range(1, STEP_N_MAX + 1)],
                        p=step_n_policy[(next_state, next_action)]
                    )
                    action_trace.append(next_action)
                    step_n_trace.append(next_step_n)

            n = step_n_trace[update_state]
            tau = t - n + 1
            if tau >= 0:
                print(len(state_trace), len(action_trace), len(reward_trace), len(step_n_trace))

                G = 0
                for i in range(tau + 1, min([tau + n, T]) + 1):
                    G += (gamma ** (i - tau - 1)) * reward_trace[i - 1]

                if tau + n < T:
                    G += (gamma ** n) * Q[state_trace[tau + n], action_trace[tau + n]]

                Q[state_trace[tau], action_trace[tau]] += alpha * (G - Q[state_trace[tau], action_trace[tau]])
                N[state_trace[tau], action_trace[tau], n] += alpha * (G - N[state_trace[tau], action_trace[tau], n])

                if learn_policy:
                    policy[state_trace[tau]] = e_greedy(env, epsilon, Q, state_trace[tau])
                    step_n_policy[(state_trace[tau], action_trace[tau])] = step_n_e_greedy(epsilon, N, state_trace[tau], action_trace[tau])

                update_state += 1

            current_state = next_state
            action = next_action

            if tau == (T - 1):
                break
            t += 1

    return policy, Q, cumulative_reward/num_iter


if __name__ == '__main__':
    cumulative_reward_lst = []

    # 그리드 월드 환경 객체 생성
    env = GridWorld(transition_reward=-0.1)

    for _ in range(1):
        policy, Q, cumulative_reward = variable_n_step_sarsa(env, epsilon=0.2, alpha=0.5, gamma=0.98, num_iter=100)
        cumulative_reward_lst.append(cumulative_reward)
    print(policy)
    print(Q)
    print("average_reward:", sum(cumulative_reward_lst)/len(cumulative_reward_lst))
