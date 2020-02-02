import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams["font.family"] = 'NanumBarunGothic'
plt.rcParams["font.size"] = 12
mpl.rcParams['axes.unicode_minus'] = False

# 0: 왼쪽 종료 상태 T1를 나타냄, 상태 가치는 0.0으로 변하지 않음
# 6: 오른쪽 종료 상태 T2를 나타냄, 상태 가치는 1.0으로 변하지 않음
# 1부터 5는 각각 차례로 상태 A부터 상태 E를 나타냄, 각 상태 가치는 0.5로 초기화됨
VALUES = np.zeros(7)
VALUES[0] = 0.0
VALUES[6] = 1.0
VALUES[1:6] = 0.5

# 올바른 상태 가치 값 저장
TRUE_VALUE = np.zeros(7)
TRUE_VALUE[0] = 0.0
TRUE_VALUE[6] = 1.0
TRUE_VALUE[1:6] = np.arange(1, 6) / 6.0

# 종료 상태를 제외한 임의의 상태에서 동일한 확률로 왼쪽 이동 또는 오른쪽 이동
LEFT_ACTION = 0
RIGHT_ACTION = 1


# @values: 현재의 상태 가치
# @alpha: 스텝 사이즈
# @batch: 배치 업데이트 유무
def temporal_difference(values, alpha=0.1, batch=False):
    state = 3
    trajectory = [state]
    rewards = [0]
    while True:
        old_state = state

        # np.random.binomial의 첫번째 인수 n은 시행 횟수를 의미, n=1 이므로 1개의 값이 샘플링됨
        if np.random.binomial(n=1, p=0.5) == LEFT_ACTION:
            state -= 1
        else:
            state += 1

        # 일단 모든 보상은 0으로 설정됨
        reward = 0
        trajectory.append(state)

        # TD 갱신 수행
        if not batch:
            values[old_state] += alpha * (reward + values[state] - values[old_state])

        if state == 6 or state == 0:
            break

        rewards.append(reward)

    if batch:
        return trajectory, rewards


# @values: 현재의 상태 가치
# @alpha: 스텝 사이즈
# @batch: 배치 업데이트 유무
def constant_alpha_monte_carlo(values, alpha=0.1, batch=False):
    state = 3
    trajectory = [state]

    # 만약 T1에서 종료되면 누적 보상(또는 이득) returns 값은 0
    # 만약 T2에서 종료되면 누적 보상(또는 이득) returns 값은 1
    while True:
        if np.random.binomial(n=1, p=0.5) == LEFT_ACTION:
            state -= 1
        else:
            state += 1

        trajectory.append(state)

        if state == 6:
            returns = 1.0
            break
        elif state == 0:
            returns = 0.0
            break

    if batch:
        return trajectory, [returns] * (len(trajectory) - 1)
    else:
        for state_ in trajectory[:-1]:
            # MC 갱신
            values[state_] += alpha * (returns - values[state_])


# 실전 연습의 왼쪽 그래프
def compute_state_values():
    episodes = [0, 1, 10, 100]
    markers = ['o', '+', 'D']
    plt.figure()
    V = np.copy(VALUES)
    plt.plot(['T1', 'A', 'B', 'C', 'D', 'E', 'T2'], V, label='초기 가치', linestyle=":")

    for i in range(1, len(episodes)):
        V = np.copy(VALUES)
        for _ in range(episodes[i]):
            temporal_difference(V)
        plt.plot(['T1', 'A', 'B', 'C', 'D', 'E', 'T2'], V, label=str(episodes[i]) + ' 에피소드', marker=markers[i-1])

    plt.plot(['T1', 'A', 'B', 'C', 'D', 'E', 'T2'], TRUE_VALUE, label='올바른 가치', linestyle="--")

    plt.xlabel('상태')
    plt.ylabel('추정 가치')
    plt.legend()
    plt.savefig('images/example_6_2_left.png')
    plt.close()


# Example 6.2 right
def rms_errors():
    # Same alpha value can appear in both arrays
    td_alphas = [0.15, 0.1, 0.05, 0.025]
    mc_alphas = [0.01, 0.02, 0.03, 0.04]
    total_runs = 100
    episodes = 100 + 1
    plt.figure()

    for i, alpha in enumerate(td_alphas + mc_alphas):
        total_errors = np.zeros(episodes)
        if i < len(td_alphas):
            method = 'TD(0)'
            linestyle = '-'
        else:
            method = 'MC'
            linestyle = '-.'

        for _ in tqdm(range(total_runs)):
            errors = []
            V = np.copy(VALUES)
            for i in range(episodes):
                errors.append(np.sqrt(np.sum(np.power(TRUE_VALUE - V, 2)) / 7.0))
                if method == 'TD(0)':
                    temporal_difference(V, alpha=alpha)
                else:
                    constant_alpha_monte_carlo(V, alpha=alpha)
            total_errors += np.asarray(errors)
        total_errors /= total_runs
        plt.plot(total_errors, linestyle=linestyle, label=method + ', alpha = {0:.2f}'.format(alpha))
    plt.xlabel('에피소드')
    plt.ylabel('RMS 에러')
    plt.legend()
    plt.savefig('images/example_6_2_right.png')
    plt.close()


# @method: 'TD(0)' 또는 'MC'
def batch_updating(method, episodes, alpha=0.001):
    # episodes 횟수의 독립적인 수행 결과 모음
    total_errors = np.zeros(episodes)

    total_runs = 100
    for _ in tqdm(range(total_runs)):
        V = np.copy(VALUES)
        errors = []

        trajectories = []
        rewards = []
        for _ in range(episodes):
            if method == 'TD(0)':
                #
                trajectory_, rewards_ = temporal_difference(V, batch=True)
            else:
                trajectory_, rewards_ = constant_alpha_monte_carlo(V, batch=True)

            trajectories.append(trajectory_)
            rewards.append(rewards_)

            while True:
                updates = np.zeros(7)
                for trajectory_, rewards_ in zip(trajectories, rewards):
                    for i in range(len(trajectory_) - 1):
                        if method == 'TD(0)':
                            updates[trajectory_[i]] += rewards_[i] + V[trajectory_[i + 1]] - V[trajectory_[i]]
                        else:
                            returns = rewards_[i]
                            updates[trajectory_[i]] += returns - V[trajectory_[i]]
                updates *= alpha
                if np.sum(np.abs(updates)) < 1e-3:
                    break
                # 배치 업데이트
                V += updates
            # RMS 에러 계산
            errors.append(np.sqrt(np.sum(np.power(V - TRUE_VALUE, 2)) / 5.0))
        total_errors += np.asarray(errors)
    total_errors /= total_runs
    return total_errors


def batch_updating_execution_episode():
    episodes = 100 + 1
    td_erros = batch_updating('TD(0)', episodes)
    mc_erros = batch_updating('MC', episodes)

    plt.plot(td_erros, label='TD(0)')
    plt.plot(mc_erros, label='MC')
    plt.xlabel('에피소드')
    plt.ylabel('RMS 에러')
    plt.legend()

    plt.savefig('images/batch_updating.png')
    plt.close()


def batch_updating_execution_alpha():
    episodes = 100 + 1
    td_erros = batch_updating('TD(0)', episodes, alpha=0.002)
    mc_erros = batch_updating('MC', episodes, alpha=0.002)

    plt.plot(td_erros, label='TD(0)')
    plt.plot(mc_erros, label='MC')
    plt.xlabel('에피소드')
    plt.ylabel('RMS 에러')
    plt.legend()

    plt.savefig('images/batch_updating_alpha.png')
    plt.close()


if __name__ == '__main__':
    # compute_state_values()
    # rms_errors()
    # batch_updating_execution_episode()
    batch_updating_execution_alpha()
