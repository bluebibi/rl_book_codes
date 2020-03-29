import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.table import Table
from environments.gridworld import GridWorld

# 이미지 저장 경로 확인 및 생성
if not os.path.exists('images/'):
    os.makedirs('images/')

GRID_HEIGHT = 5
GRID_WIDTH = 5

DISCOUNTING_RATE = 0.9      # 감쇄율

A_POSITION = (0, 1)         # 임의로 지정한 특별한 상태 A 좌표
B_POSITION = (0, 3)         # 임의로 지정한 특별한 상태 B 좌표

A_PRIME_POSITION = (4, 1)   # 상태 A에서 행동시 도착할 위치 좌표
B_PRIME_POSITION = (2, 3)   # 상태 B에서 행동시 도착할 위치 좌표

ACTION_PROBABILITY = 0.25


# 기본 GridWorld 환경에 추가할 조건을 정의하는 함수
def unique_step_wormhole(state, action):
    if state == A_POSITION:
        return A_PRIME_POSITION, 10, False, {'exec': True}
    elif state == B_POSITION:
        return B_PRIME_POSITION, 5, False, {'exec': True}
    return state, None, False, {'exec': False}


def unique_step_desert(state, action):
    x, y = state
    if action == 0:  # ACTION_UP
        x = max(x - 1, 0)
    elif action == 1:  # ACTION_DOWN
        x = min(x + 1, GRID_HEIGHT - 1)
    elif action == 2:  # ACTION_LEFT
        y = max(y - 1, 0)
    elif action == 3:  # ACTION_RIGHT
        y = min(y + 1, GRID_WIDTH - 1)

    if (x, y) == state:
        return (x, y), -1, False, {'exec': True}
    else:
        return state, None, False, {'exec': False}


# 학습 이후의 가치함수를 표 형태로 그리는 함수
def draw_image(image):
    # 축 표시 제거, 크기 조절 등 이미지 그리기 이전 설정 작업
    fig, axis = plt.subplots()
    axis.set_axis_off()
    table = Table(axis, bbox=[0, 0, 1, 1])

    num_rows, num_cols = image.shape
    width, height = 1.0 / num_cols, 1.0 / num_rows

    # 렌더링할 이미지에 표 셀 추가
    for (i, j), val in np.ndenumerate(image):
        if val == 0.0:
            val = ''
        table.add_cell(i, j, width, height, text=val, loc='center', facecolor='white')

    # 행, 열 라벨 추가
    for i in range(len(image)):
        table.add_cell(i, -1, width, height, text=i+1, loc='right', edgecolor='none', facecolor='none')
        table.add_cell(-1, i, width, height/2, text=i+1, loc='center', edgecolor='none', facecolor='none')

    for key, cell in table.get_celld().items():
         cell.get_text().set_fontsize(20)

    axis.add_table(table)


# GRID_WORLD 첫번째 예제
def grid_world_state_values():
    # 모든 값이 0으로 채워진 5x5 맵 생성, 가치 함수로 해석
    value_function = np.zeros((GRID_HEIGHT, GRID_WIDTH))

    # 가치 함수의 값들이 수렴할 때까지 반복
    while True:
        # value_function과 동일한 형태를 가지면서 값은 모두 0인 배열을 new_value_function에 저장
        new_value_function = np.zeros_like(value_function)
        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                env = GridWorld(height=GRID_HEIGHT, width=GRID_WIDTH, start_state=(i, j), terminal_state=[], transition_reward=0, terminal_reward=0,
                                unique_steps=[unique_step_wormhole, unique_step_desert])

                # 현재 상태에서 가능한 모든 행동들의 결과로 다음 상태들을 갱신
                for action in env.observation_space.ACTIONS:
                    env.reset()
                    (next_i, next_j), reward, _, _ = env.step(action)
                    # Bellman-Equation, 벨만 방정식 적용
                    # 모든 행동에 대해 그 행동의 확률, 행동 이후의 누적 기대 보상을 갱신에 사용
                    new_value_function[i, j] += ACTION_PROBABILITY * (reward + DISCOUNTING_RATE * value_function[next_i, next_j])

        # 가치 함수 수렴 여부 판단
        if np.sum(np.abs(value_function - new_value_function)) < 1e-4:
            draw_image(np.round(new_value_function, decimals=2))
            plt.savefig('images/grid_world_state_values.png')
            plt.close()
            break

        # 가치 함수 갱신
        value_function = new_value_function


# GRID_WORLD 두번째 예제
def grid_world_optimal_values():
    # 모든 값이 0으로 채워진 5x5 맵 생성, 가치 함수로 해석
    value_function = np.zeros((GRID_HEIGHT, GRID_WIDTH))

    # 가치 함수의 값들이 수렴할 때까지 반복
    while True:
        new_value_function = np.zeros_like(value_function)
        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                env = GridWorld(height=GRID_HEIGHT, width=GRID_WIDTH, start_state=(i, j), terminal_state=[], transition_reward=0, terminal_reward=0,
                                unique_steps=[unique_step_wormhole, unique_step_desert])
                values = []
                for action in env.observation_space.ACTIONS:
                    env.reset()
                    (next_i, next_j), reward, _, _ = env.step(action)
                    # Value-Iteration 기법 적용
                    values.append(reward + DISCOUNTING_RATE * value_function[next_i, next_j])

                # 새롭게 계산된 상태 가치 중 최대 상태 가치로 현재 상태의 가치로 갱신
                new_value_function[i, j] = np.max(values)

        # 가치 함수 수렴 여부 판단
        if np.sum(np.abs(new_value_function - value_function)) < 1e-4:
            draw_image(np.round(new_value_function, decimals=2))
            plt.savefig('images/grid_world_optimal_values.png')
            plt.close()
            break

        value_function = new_value_function


# MAIN
if __name__ == '__main__':
    value_function = np.zeros((GRID_HEIGHT, GRID_WIDTH))
    draw_image(np.round(value_function, decimals=0))
    plt.savefig('images/empty_grid_world.png')
    plt.close()

    # GRID_WORLD 실행
    grid_world_state_values()
    grid_world_optimal_values()
