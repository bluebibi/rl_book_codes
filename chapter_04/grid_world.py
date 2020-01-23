import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.table import Table

# 학습 결과를 렌더링할 방법을 지정
matplotlib.use('Agg')

WORLD_SIZE = 4
# 상, 좌, 하, 우 이동을 나타내는 배열
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]
ACTION_PROBABILITY = 0.25


# 종료 상태인지 검사하는 함수
def is_terminal(state):
    x, y = state
    return (x == 0 and y == 0) or (x == WORLD_SIZE - 1 and y == WORLD_SIZE - 1)


# 에이전트가 상태 state(t) 에서 행동 action을 했을 때의 결과로 나타나는
# 다음 상태 next_state와 보상 reward를 반환하는 함수
def step(state, action):
    if is_terminal(state):
        return state, 0

    # ACTIONS 배열에서 행동에 따른 x, y축 이동을 값으로 저장했으므로
    # 현재 상태(x, y 좌표)에서 해당 값을 더해주는 것으로 다음 상태를 표현
    next_state = (np.array(state) + action).tolist()
    x, y = next_state

    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        next_state = state

    # 이동에 드는 비용은 -1로 고정
    reward = -1
    return next_state, reward


# 학습 이후의 가치함수를 표 형태로 그리는 함수
def draw_image(image):
    # 축 표시 제거, 크기 조절 등 이미지 그리기 이전 설정 작업
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # 렌더링 할 이미지에 표 셀과 해당 값 추가
    for (i, j), val in np.ndenumerate(image):
        tb.add_cell(i, j, width, height, text=val,
                    loc='center', facecolor='white')

    # 행, 열 라벨 추가
    for i in range(len(image)):
        tb.add_cell(i, -1, width, height, text=i+1, loc='right',
                    edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height/2, text=i+1, loc='center',
                    edgecolor='none', facecolor='none')

    ax.add_table(tb)


# 상태 가치 함수를 계산하는 함수
def compute_state_value(in_place=True, discounting_rate=1.0):
    # 모든 값이 0으로 채워진 4x4 맵 생성, 상태 가치 함수로 해석
    new_state_values = np.zeros((WORLD_SIZE, WORLD_SIZE))
    iteration = 0

    # 가치 함수의 값들이 수렴할 때까지 반복
    while True:
        # in-place, out-place 전략 구분
        # 과정에서의 미세한 차이는 보이나 결과는 동일하게 나타남
        if in_place:
            state_values = new_state_values
        else:
            state_values = new_state_values.copy()
        old_state_values = state_values.copy()

        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                value = 0
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    # 모든 행동에 대해 그 행동의 확률, 행동 이후의 누적 기대 보상을 갱신에 사용
                    value += ACTION_PROBABILITY * (reward + discounting_rate * state_values[next_i, next_j])
                new_state_values[i, j] = value

        # 갱신되는 값이 0.0001을 기준으로 수렴하는지 판정
        max_delta_value = abs(old_state_values - new_state_values).max()
        if max_delta_value < 1e-4:
            break

        iteration += 1

    return new_state_values, iteration


def figure_4_1():
    # 동일 장소(in-place) 전략 기반으로 생성한 가치 함수와
    # 비동일 장소(out-place)으로 생성한 가치 함수가 결국 동일하게 수렴함을 보이기 위해
    # 각각의 경우를 수행하고 결과를 이미지로 저장
    in_place_values, in_place_iteration = compute_state_value(in_place=True)
    out_place_values, out_place_iteration = compute_state_value(in_place=False)

    print('in-place  전략: {} 회 반복'.format(in_place_iteration))
    print('out-place 전략: {} 회 반복'.format(out_place_iteration))

    # 이미지 저장 경로 존재 여부 확인
    if not os.path.exists('images/'):
        os.makedirs('images/')

    # in-place 결과 저장
    draw_image(np.round(in_place_values, decimals=2))
    plt.savefig('images/in_place_values.png')
    plt.close()
    # out-place 결과 저장
    draw_image(np.round(out_place_values, decimals=2))
    plt.savefig('images/out_place_values.png')
    plt.close()


# MAIN
if __name__ == '__main__':
    figure_4_1()