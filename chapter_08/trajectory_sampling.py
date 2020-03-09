import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams["font.family"] = 'NanumBarunGothic'
plt.rcParams["font.size"] = 12
mpl.rcParams['axes.unicode_minus'] = False

# 2개의 행동
ACTIONS = [0, 1]

# 각 전이를 수행할 때 종료 상태로 전이될 확률: 0.1
TERMINATION_PROB = 0.1

# 최대 갱신 횟수
MAX_STEPS = 20000

# epsilon-탐욕 행동 정책의 파라미터
EPSILON = 0.1


# max_q가 여러 개 존재할 경우를 고려한 argmax 함수
def argmax(value):
    max_q = np.max(value)
    return np.random.choice([a for a, q in enumerate(value) if q == max_q])


class Task:
    # @n_states: 종료 상태를 제외한 상태의 개수
    # @b: 브렌칭 요소
    # 각 에피소드는 상태 0에서 시작하고, 상태 n_states는 종료 상태
    def __init__(self, n_states, b):
        self.n_states = n_states
        self.b = b

        # transition matrix, each state-action pair leads to b possible states
        self.transition = np.random.randint(n_states, size=(n_states, len(ACTIONS), b))

        # it is not clear how to set the reward, I use a unit normal distribution here
        # reward is determined by (s, a, s')
        self.reward = np.random.randn(n_states, len(ACTIONS), b)

    # 현재 상태 state 와 행동 action을 받아서 보상과 다음 상태를 반환
    def step(self, state, action):
        if np.random.rand() < TERMINATION_PROB:
            return 0, self.n_states

        next = np.random.randint(self.b)

        return self.reward[state, action, next], self.transition[state, action, next]


# 탐욕적 정책하에서 초기 상태의 가치를 평가
# derived from @q under the MDP @task
def evaluate_pi(q, task):
    # use Monte Carlo method to estimate the state value
    runs = 1000
    returns = []
    for r in range(runs):
        rewards = 0
        state = 0
        while state < task.n_states:
            action = argmax(q[state])
            reward, state = task.step(state, action)
            rewards += reward
        returns.append(rewards)
    return np.mean(returns)


# perform expected update from a uniform state-action distribution of the MDP @task
# evaluate the learned q value every @eval_interval steps
def uniform(task, eval_interval):
    performance = []
    q = np.zeros((task.n_states, 2))
    for step in tqdm(range(MAX_STEPS)):
        state = step // len(ACTIONS) % task.n_states
        action = step % len(ACTIONS)

        next_states = task.transition[state, action]
        q[state, action] = (1 - TERMINATION_PROB) * np.mean(task.reward[state, action] + np.max(q[next_states, :], axis=1))

        if step % eval_interval == 0:
            v_pi = evaluate_pi(q, task)
            performance.append([step, v_pi])

    return zip(*performance)


# perform expected update from an on-policy distribution of the MDP @task
# evaluate the learned q value every @eval_interval steps
def on_policy(task, eval_interval):
    performance = []
    q = np.zeros((task.n_states, 2))
    state = 0
    for step in tqdm(range(MAX_STEPS)):
        if np.random.rand() < EPSILON:
            action = np.random.choice(ACTIONS)
        else:
            action = argmax(q[state])

        _, next_state = task.step(state, action)

        next_states = task.transition[state, action]
        q[state, action] = (1 - TERMINATION_PROB) * np.mean(task.reward[state, action] + np.max(q[next_states, :], axis=1))

        if next_state == task.n_states:
            next_state = 0

        state = next_state

        if step % eval_interval == 0:
            v_pi = evaluate_pi(q, task)
            performance.append([step, v_pi])

    return zip(*performance)


def trajectory_sampling():
    num_states_list = [1000, 10000]
    num_branches_list = [1, 3, 10]
    methods = [on_policy, uniform]

    # average across 30 tasks
    n_tasks = 30

    # number of evaluation points
    x_ticks = 100

    plt.figure(figsize=(10, 20))
    for i, num_states in enumerate(num_states_list):
        plt.subplot(2, 1, i + 1)
        for num_branches in num_branches_list:
            tasks = [Task(num_states, num_branches) for _ in range(n_tasks)]
            for method in methods:
                value = []
                for task in tasks:
                    steps, v = method(task, MAX_STEPS / x_ticks)
                    value.append(v)
                value = np.mean(np.asarray(value), axis=0)
                plt.plot(steps, value, label='b = {0}, {1}'.format(num_branches, method.__name__))
        plt.title('총 상태의 개수: {0}'.format(num_states))

        plt.ylabel('초기 상태 가치')
        plt.legend()

    plt.subplot(2, 1, 2)
    plt.xlabel('갱신 횟수')

    plt.savefig('images/trajectory_sampling.png')
    plt.close()


if __name__ == '__main__':
    trajectory_sampling()