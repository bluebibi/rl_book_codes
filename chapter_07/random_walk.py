import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams["font.family"] = 'NanumBarunGothic'
plt.rcParams["font.size"] = 12
mpl.rcParams['axes.unicode_minus'] = False

# 모든 상태의 개수 (종료 상태 제외)
N_STATES = 19

# 감가율
GAMMA = 1

# 종료 상태를 제외한 모든 상태
STATES = np.arange(1, N_STATES + 1)

# 초기 상태 지정
START_STATE = 10

# 두 개의 종료 상태 지정
# 맨 왼쪽의 종료 상태(0)으로 이동하는 행동은 -1의 보상 발생
# 맨 오른쪽의 종료 상태(20)으로 이동하는 행동은 1의 보상 발생
TERMINAL_STATES = [0, N_STATES + 1]

# 벨만 방정식으로 유도된 올바른 상태 가치 값
TRUE_VALUE = np.arange(-20, 22, 2) / 20.0
TRUE_VALUE[0] = TRUE_VALUE[-1] = 0
print(TRUE_VALUE)


# n-스텝 TD 방법
# @value: 본 함수에 의하여 갱신될 각 상태의 가치
# @n: n-스텝 TD 방법의 n
# @alpha: 스텝 사이즈
def temporal_difference(value, n, alpha):
    # 초기 상태 지정
    state = START_STATE

    # 에피소드에 대하여 상태와 보상을 저장하는 배열
    states = [state]
    rewards = [0]

    # 타임 스텝
    time_step = 0

    # 이 에피소드의 길이
    T = float('inf')
    while True:
        # 다음 타임 스텝
        time_step += 1

        if time_step < T:
            # 행동을 임의로 선택하여 다음 상태 결정
            if np.random.binomial(1, 0.5) == 1:
                next_state = state + 1
            else:
                next_state = state - 1

            if next_state == 0:
                reward = -1
            elif next_state == 20:
                reward = 1
            else:
                reward = 0

            # 다음 타임 스텝에서의 보상과 상태 저장
            rewards.append(reward)
            states.append(next_state)

            if next_state in TERMINAL_STATES:
                T = time_step

        # 갱신을 수행할 타임 스텝 결정
        tau = time_step - n

        if tau >= 0:
            returns = 0.0

            # 대응되는 누적 보상(또는 이득)
            for i in range(tau + 1, min(tau + n, T) + 1):
                returns += pow(GAMMA, i - tau - 1) * rewards[i]

            # 누적 보상(또는 이득)에 상태 가치 추가
            if tau + n <= T:
                returns += pow(GAMMA, n) * value[states[(tau + n)]]

            state_to_update = states[tau]

            # 상태 가치 갱신
            if state_to_update not in TERMINAL_STATES:
                value[state_to_update] += alpha * (returns - value[state_to_update])

        if tau == T - 1:
            break

        state = next_state


def n_step_td_for_random_walk():
    # 가능한 스텝
    steps = np.power(2, np.arange(0, 10))

    # 가능한 스텝 사이즈
    alphas = np.arange(0, 1.1, 0.1)

    # 총 10번의 수행
    runs = 100

    # 가 수행당 10번의 에피소드 수행
    episodes = 10

    # 각 (상태, 스텝 사이즈) 쌍에 대래 오차를 추적함
    errors = np.zeros((len(steps), len(alphas)))
    for run in tqdm(range(runs)):
        for step_ind, step in enumerate(steps):
            for alpha_ind, alpha in enumerate(alphas):
                print('run:', run, ', step:', step, ', alpha:', alpha)
                value = np.zeros(N_STATES + 2)
                for ep in range(episodes):
                    temporal_difference(value, step, alpha)
                    # RMS (Rooted Mean Square) 오차 계산
                    errors[step_ind, alpha_ind] += np.sqrt(np.sum(np.power(value - TRUE_VALUE, 2)) / N_STATES)

    # RMS 오차 평균값 계산
    errors /= episodes * runs

    marker = ['o', 'x', '.', 's', '*', '+', '|', '^', 'D', ' ']
    for i in range(0, len(steps)):
        plt.plot(alphas, errors[i, :], marker=marker[i], label='n = %d' % (steps[i]))

    plt.xlabel('스텝 사이즈(alpha)')
    plt.ylabel('RMS 오차')
    plt.ylim([0.25, 0.6])
    plt.legend()

    plt.savefig('images/n_step_td_for_random_walk.png')
    plt.close()


if __name__ == '__main__':
    n_step_td_for_random_walk()