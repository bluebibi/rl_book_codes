import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams["font.family"] = 'NanumBarunGothic'
plt.rcParams["font.size"] = 12
mpl.rcParams['axes.unicode_minus'] = False

# 그리드월드 높이와 너비
WORLD_HEIGHT = 4
WORLD_WIDTH = 12

# 탐색 확률
EPSILON = 0.1

# 스텝 사이즈
ALPHA = 0.5

# 감가율
GAMMA = 1

# 모든 행동 정의
UP_ACTION = 0
DOWN_ACTION = 1
LEFT_ACTION = 2
RIGHT_ACTION = 3
ACTIONS = [UP_ACTION, DOWN_ACTION, LEFT_ACTION, RIGHT_ACTION]

# 초기 상태와 종료 상태
START_STATE = [3, 0]
TERMINAL_STATE = [3, 11]


def step(state, action):
    i, j = state
    if action == UP_ACTION:
        next_state = [max(i - 1, 0), j]
    elif action == LEFT_ACTION:
        next_state = [i, max(j - 1, 0)]
    elif action == RIGHT_ACTION:
        next_state = [i, min(j + 1, WORLD_WIDTH - 1)]
    elif action == DOWN_ACTION:
        next_state = [min(i + 1, WORLD_HEIGHT - 1), j]
    else:
        assert False

    if (state == START_STATE and action == RIGHT_ACTION) or (i == 2 and 1 <= j <= 10 and action == DOWN_ACTION):
        reward = -100
        next_state = START_STATE
    else:
        reward = -1

    return next_state, reward

# reward for each action in each state
# actionRewards = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
# actionRewards[:, :, :] = -1.0
# actionRewards[2, 1:11, DOWN_ACTION] = -100.0
# actionRewards[3, 0, RIGHT_ACTION] = -100.0

# set up destinations for each action in each state
# actionDestination = []
# for i in range(0, WORLD_HEIGHT):
#     actionDestination.append([])
#     for j in range(0, WORLD_WIDTH):
#         destinaion = dict()
#         destinaion[UP_ACTION] = [max(i - 1, 0), j]
#         destinaion[LEFT_ACTION] = [i, max(j - 1, 0)]
#         destinaion[RIGHT_ACTION] = [i, min(j + 1, WORLD_WIDTH - 1)]
#         if i == 2 and 1 <= j <= 10:
#             destinaion[DOWN_ACTION] = START_STATE
#         else:
#             destinaion[DOWN_ACTION] = [min(i + 1, WORLD_HEIGHT - 1), j]
#         actionDestination[-1].append(destinaion)
# actionDestination[3][0][RIGHT_ACTION] = START_STATE


# epsilon-탐욕적 정책에 따른 행동 선택
def choose_action(state, q_value):
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(ACTIONS)
    else:
        values_ = q_value[state[0], state[1], :]
        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])


# @q_value: 갱신해야할 행동 가치
# @step_size: 스텝 사이즈
# @expected: 이 인수가 True이면 기대값 기반 SARSA 알고리즘 수행
# @return: 본 에피소드에서의 누적 보상
def sarsa(q_value, expected=False, step_size=ALPHA):
    state = START_STATE
    action = choose_action(state, q_value)
    sum_of_rewards = 0.0

    while state != TERMINAL_STATE:
        next_state, reward = step(state, action)
        next_action = choose_action(next_state, q_value)
        sum_of_rewards += reward
        if not expected:
            expectation = q_value[next_state[0], next_state[1], next_action]
        else:
            # 새로운 상태에 대한 기대값 계산
            expectation = 0.0
            q_next = q_value[next_state[0], next_state[1], :]
            best_actions = np.argwhere(q_next == np.max(q_next))
            for action_ in ACTIONS:
                if action_ in best_actions:
                    expectation += ((1.0 - EPSILON) / len(best_actions) + EPSILON / len(ACTIONS)) * q_value[next_state[0], next_state[1], action_]
                else:
                    expectation += EPSILON / len(ACTIONS) * q_value[next_state[0], next_state[1], action_]

        q_value[state[0], state[1], action] += step_size * (reward + GAMMA * expectation - q_value[state[0], state[1], action])
        state = next_state
        action = next_action
    return sum_of_rewards


# @q_value: 갱신해야할 행동 가치
# @step_size: 스텝 사이즈
# @return: 본 에피소드에서의 누적 보상
def q_learning(q_value, step_size=ALPHA):
    state = START_STATE
    sum_of_rewards = 0.0
    while state != TERMINAL_STATE:
        action = choose_action(state, q_value)
        next_state, reward = step(state, action)
        sum_of_rewards += reward

        # Q-러닝 갱신
        max_value_for_next_state = np.max(q_value[next_state[0], next_state[1], :])
        q_value[state[0], state[1], action] += step_size * (reward + GAMMA * max_value_for_next_state - q_value[state[0], state[1], action])
        state = next_state
    return sum_of_rewards


# print optimal policy
def print_optimal_policy(q_value):
    optimal_policy = []
    for i in range(0, WORLD_HEIGHT):
        optimal_policy.append([])
        for j in range(0, WORLD_WIDTH):
            if [i, j] == TERMINAL_STATE:
                optimal_policy[-1].append('G')
                continue
            best_action = np.argmax(q_value[i, j, :])
            if best_action == UP_ACTION:
                optimal_policy[-1].append('U')
            elif best_action == DOWN_ACTION:
                optimal_policy[-1].append('D')
            elif best_action == LEFT_ACTION:
                optimal_policy[-1].append('L')
            elif best_action == RIGHT_ACTION:
                optimal_policy[-1].append('R')

    for row in optimal_policy:
        print(row)


def cumulative_rewards_for_episodes():
    # 각 수행에서 수행하는 에피소드 개수
    episodes = 500

    # 50번의 수행
    runs = 50

    rewards_sarsa = np.zeros(episodes)
    rewards_q_learning = np.zeros(episodes)
    rewards_expected_sarsa = np.zeros(episodes)

    for _ in tqdm(range(runs)):
        q_table_sarsa = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
        q_table_q_learning = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
        q_table_expected_sarsa = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))

        for i in range(episodes):
            rewards_sarsa[i] += sarsa(q_table_sarsa)
            rewards_q_learning[i] += q_learning(q_table_q_learning)
            rewards_expected_sarsa[i] += sarsa(q_table_expected_sarsa, expected=True)

    # 50번의 수행에 대해 평균 계산
    rewards_sarsa /= runs
    rewards_q_learning /= runs
    rewards_expected_sarsa /= runs

    # 그래프 출력
    plt.plot(rewards_sarsa, linestyle='-', label='SARSA')
    plt.plot(rewards_q_learning, linestyle=':', label='Q-러닝')
    plt.plot(rewards_expected_sarsa, linestyle='-.', label='기대값 기반 SARSA')
    plt.xlabel('에피소드')
    plt.ylabel('에피소드 당 누적 보상')
    plt.ylim([-100, 0])
    plt.legend()

    plt.savefig('images/cumulative_rewards_for_episodes.png')
    plt.close()

    # display optimal policy
    print('SARSA 최적 정책:')
    print_optimal_policy(q_table_sarsa)

    print('Q-러닝 최적 정책:')
    print_optimal_policy(q_table_q_learning)

    print('기대값 기반 SARSA 최적 정책:')
    print_optimal_policy(q_table_expected_sarsa)


def cumulative_rewards_for_step_size():
    # 각 수행에서 수행하는 에피소드 개수
    episodes = 1000

    # 10번의 수행
    runs = 10

    step_sizes = np.arange(0.1, 1.1, 0.1)

    ASYMPTOTIC_SARSA = 0
    ASYMPTOTIC_EXPECTED_SARSA = 1
    ASYMPTOTIC_QLEARNING = 2

    INTERIM_SARSA = 3
    INTERIM_EXPECTED_SARSA = 4
    INTERIM_QLEARNING = 5

    methods = range(0, 6)

    performace = np.zeros((6, len(step_sizes)))

    for run in range(runs):
        for idx, step_size in tqdm(list(zip(range(len(step_sizes)), step_sizes))):
            q_table_sarsa = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
            q_table_expected_sarsa = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
            q_table_q_learning = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))

            for episode in range(episodes):
                sarsa_reward = sarsa(q_table_sarsa, expected=False, step_size=step_size)
                expected_sarsa_reward = sarsa(q_table_expected_sarsa, expected=True, step_size=step_size)
                q_learning_reward = q_learning(q_table_q_learning, step_size=step_size)

                performace[ASYMPTOTIC_SARSA, idx] += sarsa_reward
                performace[ASYMPTOTIC_EXPECTED_SARSA, idx] += expected_sarsa_reward
                performace[ASYMPTOTIC_QLEARNING, idx] += q_learning_reward

                if episode < 100:
                    performace[INTERIM_SARSA, idx] += sarsa_reward
                    performace[INTERIM_EXPECTED_SARSA, idx] += expected_sarsa_reward
                    performace[INTERIM_QLEARNING, idx] += q_learning_reward

    performace[:3, :] /= episodes * runs
    performace[3:, :] /= 100 * runs
    labels = [
        'SARSA (1000 에피소드)', '기대값 기반 SARS (1000 에피소드)', 'Q-러닝 (1000 에피소드)',
        'SARSA (100 에피소드)', '기대값 기반 SARSA (100 에피소드)', 'Q-러닝 (100 에피소드)'
    ]

    for method, label in zip(methods, labels):
        if method == 0 or method == 1 or method == 2:
            linestyle = '-'
        else:
            linestyle = ':'

        if method == 0 or method == 3:
            marker = 'o'
        elif method == 1 or method == 4:
            marker = 'x'
        else:
            marker = '+'

        plt.plot(step_sizes, performace[method, :], linestyle=linestyle, marker=marker, label=label)
        
    plt.xlabel('스텝 사이즈 (alpha)')
    plt.ylabel('에피소드 당 누적 보상')
    plt.legend()

    plt.savefig('images/cumulative_rewards_for_step_size.png')
    plt.close()


if __name__ == '__main__':
    cumulative_rewards_for_episodes()
    cumulative_rewards_for_step_size()