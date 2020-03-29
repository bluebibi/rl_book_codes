import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.table import Table
from environments.gridworld import GridWorld

GRID_HEIGHT = 4
GRID_WIDTH = 4
TERMINAL_STATE = [(0, 0), (GRID_HEIGHT-1, GRID_WIDTH-1)]
ACTION_PROBABILITY = 0.25


# 학습 이후의 가치함수를 표 형태로 그리는 함수
def draw_image(image):
    # 축 표시 제거, 크기 조절 등 이미지 그리기 이전 설정 작업
    fig, ax = plt.subplots()
    ax.set_axis_off()
    table = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # 렌더링 할 이미지에 표 셀과 해당 값 추가
    for (i, j), val in np.ndenumerate(image):
        table.add_cell(i, j, width, height, text=val, loc='center', facecolor='white')

    # 행, 열 라벨 추가
    for i in range(len(image)):
        table.add_cell(i, -1, width, height, text=i+1, loc='right', edgecolor='none', facecolor='none')
        table.add_cell(-1, i, width, height/2, text=i+1, loc='center', edgecolor='none', facecolor='none')

    for key, cell in table.get_celld().items():
         cell.get_text().set_fontsize(20)

    ax.add_table(table)


# 상태 가치 함수를 계산하는 함수
def compute_state_value(in_place=True, discounting_rate=1.0):
    # 모든 값이 0으로 채워진 4x4 맵 생성, 상태 가치 함수로 해석
    new_state_values = np.zeros((GRID_HEIGHT, GRID_WIDTH))
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

        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                env = GridWorld(height=GRID_HEIGHT, width=GRID_WIDTH, start_state=(i, j),
                                terminal_state=TERMINAL_STATE, transition_reward=-1, terminal_reward=-1)

                value = 0
                for action in env.observation_space.ACTIONS:
                    env.reset()
                    (next_i, next_j), reward, done, _ = ((i, j), 0, None, None) if (i, j) in TERMINAL_STATE else env.step(action)

                    # 모든 행동에 대해 그 행동의 확률, 행동 이후의 누적 기대 보상을 상태 가치 갱신에 사용
                    value += ACTION_PROBABILITY * (reward + discounting_rate * state_values[next_i, next_j])
                new_state_values[i, j] = value

        # 갱신되는 값이 0.0001을 기준으로 수렴하는지 판정
        max_delta_value = abs(old_state_values - new_state_values).max()
        if max_delta_value < 1e-4:
            break

        iteration += 1

    return new_state_values, iteration


def grid_world_policy_improvement():
    # 동일 장소(in-place) 전략 기반으로 생성한 가치 함수와
    # 비동일 장소(out-place)으로 생성한 가치 함수가 결국 동일하게 수렴함을 보이기 위해
    # 각각의 경우를 수행하고 결과를 이미지로 저장
    in_place_values, in_place_iteration = compute_state_value(in_place=True)
    out_place_values, out_place_iteration = compute_state_value(in_place=False)

    print('in-place  전략: {} 회 반복'.format(in_place_iteration))
    print('out-place 전략: {} 회 반복'.format(out_place_iteration))

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
    # 이미지 저장 경로 확인 및 생성
    if not os.path.exists('images/'):
        os.makedirs('images/')

    grid_world_policy_improvement()
