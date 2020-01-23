import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.table import Table

# 학습 결과를 렌더링할 방법을 지정
matplotlib.use('Agg')

WORLD_SIZE = 5
DISCOUNTING_RATE = 0.9      # 감쇄율
A_POSITION = [0, 1]         # 임의로 지정한 특별한 상태 A 좌표
B_POSITION = [0, 3]         # 임의로 지정한 특별한 상태 B 좌표
A_PRIME_POSITION = [4, 1]   # 상태 A에서 행동시 도착할 위치 좌표
B_PRIME_POSITION = [2, 3]   # 상태 B에서 행동시 도착할 위치 좌표

# 좌, 상, 우, 하 이동을 나타내는 배열
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]
ACTION_PROBABILITY = 0.25


# 에이전트가 상태 S(t)에서 행동 A(action)을 했을 때의 결과로 나타나는
# 다음 상태 S(t+1)과 보상 R(t+1)을 반환하는 함수
def step(state, action):
    # A, B 위치에서의 행동은 항상 A', B' 위치로 이동시키고 고정된 보상을 받는다.
    if state == A_POSITION:
        return A_PRIME_POSITION, 10
    if state == B_POSITION:
        return B_PRIME_POSITION, 5

    # ACTIONS 배열에서 행동에 따른 x, y축 이동을 값으로 저장했으므로
    # 현재 상태(x, y 좌표)에서 해당 값을 더해주는 것으로 다음 상태를 표현
    next_state = (np.array(state) + action).tolist()
    x, y = next_state

    # 맵 밖으로 나간 경우 -1의 보상을 받고 상태 유지
    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        reward = -1.0
        next_state = state
    else:
        reward = 0

    return next_state, reward


# 학습 이후의 가치함수를 표 형태로 그리는 함수
def draw_image(image):
    # 축 표시 제거, 크기 조절 등 이미지 그리기 이전 설정 작업
    figure, axis = plt.subplots()
    axis.set_axis_off()
    table = Table(axis, bbox=[0, 0, 1, 1])

    num_rows, num_cols = image.shape
    width, height = 1.0 / num_cols, 1.0 / num_rows

    # 렌더링할 이미지에 표 셀 추가
    for (i, j), val in np.ndenumerate(image):
        table.add_cell(i, j, width, height, text=val,
                    loc='center', facecolor='white')

    # 행, 열 라벨 추가
    for i in range(len(image)):
        table.add_cell(i, -1, width, height, text=i+1, loc='right',
                    edgecolor='none', facecolor='none')
        table.add_cell(-1, i, width, height/2, text=i+1, loc='center',
                    edgecolor='none', facecolor='none')

    axis.add_table(table)


# GRID_WORLD 첫번째 예제
def grid_world_bellman_equation():
    # 모든 값이 0으로 채워진 5x5 맵 생성, 가치 함수로 해석
    value_function = np.zeros((WORLD_SIZE, WORLD_SIZE))

    # 가치 함수의 값들이 수렴할 때까지 반복
    while True:
        # value_function과 동일한 형태를 가지면서 값은 모두 0인 배열을 new_value_function에 저장
        new_value_function = np.zeros_like(value_function)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                # 현재 상태에서 가능한 모든 행동들의 결과로 다음 상태들을 갱신
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    # Bellman-Equation, 벨만 방정식 적용
                    # 모든 행동에 대해 그 행동의 확률, 행동 이후의 누적 기대 보상을 갱신에 사용
                    new_value_function[i, j] += ACTION_PROBABILITY * (reward + DISCOUNTING_RATE * value_function[next_i, next_j])

        # 가치 함수 수렴 여부 판단
        if np.sum(np.abs(value_function - new_value_function)) < 1e-4:
            draw_image(np.round(new_value_function, decimals=2))
            plt.savefig('images/grid_world_bellman_equation.png')
            plt.close()
            break

        # 가치 함수 갱신
        value_function = new_value_function


# GRID_WORLD 두번째 예제
def grid_world_value_iteration():
    # 모든 값이 0으로 채워진 5x5 맵  생성, 가치 함수로 해석
    value_function = np.zeros((WORLD_SIZE, WORLD_SIZE))

    # 가치 함수의 값들이 수렴할 때까지 반복
    while True:
        new_value_function = np.zeros_like(value_function)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                values = []
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    # Value-Iteration 기법 적용
                    values.append(reward + DISCOUNTING_RATE * value_function[next_i, next_j])
                # 현재 상태에서 가능한 수들 중 가장 기대 보상이 큰 경우만을 갱신에 사용
                new_value_function[i, j] = np.max(values)

        # 가치 함수 수렴 여부 판단
        if np.sum(np.abs(new_value_function - value_function)) < 1e-4:
            draw_image(np.round(new_value_function, decimals=2))
            plt.savefig('images/grid_world_value_iteration.png')
            plt.close()
            break

        value_function = new_value_function


# MAIN
if __name__ == '__main__':
    # 이미지 저장 경로 확인 및 생성
    if not os.path.exists('images/'):
        os.makedirs('images/')

    # GRID_WORLD 실행
    grid_world_bellman_equation()
    grid_world_value_iteration()