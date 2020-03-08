import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

# 이미지 저장 경로 확인 및 생성
from environments.tic_tac_toe import TicTacToe, INITIAL_STATE, ALL_STATES, PLAYER_1, PLAYER_2, generate_all_states, \
    get_new_state, BOARD_ROWS, BOARD_COLS

if not os.path.exists('images/'):
    os.makedirs('images/')

plt.rcParams["font.family"] = 'NanumBarunGothic'
plt.rcParams["font.size"] = 12
mpl.rcParams['axes.unicode_minus'] = False


class Player:
    def __init__(self, player_int, step_size=0.1, epsilon=0.1):
        # Player_1(선공), Player_2(후공) 구분을 위한 값
        self.player_int = player_int

        # 특정 상태에서의 가치 함수들의 집합
        self.estimated_values = dict()

        # 가치 함수 갱신 비율
        self.step_size = step_size

        # 탐욕적 방법으로 행동하지 않을 확률
        self.epsilon = epsilon
        self.visited_states = []

    def reset(self):
        self.visited_states.clear()
        self.append_state(INITIAL_STATE)

    def append_state(self, state):
        self.visited_states.append(state)

    def initialize_estimated_values(self):
        # 가치 함수 초기화
        for state_identifier in ALL_STATES:
            state, is_end = ALL_STATES[state_identifier]
            if is_end:
                if state.winner == self.player_int:
                    self.estimated_values[state_identifier] = 1.0
                elif state.winner == 0:
                    self.estimated_values[state_identifier] = 0.5
                else:
                    self.estimated_values[state_identifier] = 0
            else:
                self.estimated_values[state_identifier] = 0.5

    # 게임 1회 종료 후 가치 함수 갱신
    def update_estimated_values(self):
        states_hash_values = [state.identifier() for state in self.visited_states]

        # 게임 처음 상태부터 마지막 상태까지의 역순으로
        for i in reversed(range(len(states_hash_values) - 1)):
            state_hash_value = states_hash_values[i]
            next_state_hash_value = states_hash_values[i + 1]

            # 행동 이후 상태와 이전 상태 기댓값의 차이 계산
            temporal_difference = self.estimated_values[next_state_hash_value] - self.estimated_values[state_hash_value]

            # 해당 상태의 가치 함수를 갱신
            # 공식 V(S(t)) <- V(S(t)) + a[V(S(t+1)) - V(S(t))], a = step_size
            self.estimated_values[state_hash_value] += self.step_size * temporal_difference

    # 현재 상태에 기반하여 행동 결정
    def act(self):
        state = self.visited_states[-1]

        # 현재 상태에서 도달 가능한 다음 상태들을 저장
        possible_states = []
        possible_positions = []

        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if state.data[i, j] == 0:
                    possible_positions.append([i, j])
                    new_state = get_new_state(i, j, state.data, self.player_int)
                    new_state_hash = new_state.identifier()

                    if new_state_hash not in ALL_STATES:
                        print(self.player_int, new_state, new_state_hash)

                    assert new_state_hash in ALL_STATES
                    possible_states.append(new_state_hash)

        # epsilon 값에 의해 확률적으로 임의의 행동 수행
        if np.random.rand() < self.epsilon:
            i, j = possible_positions[np.random.randint(len(possible_positions))]
            return i, j

        next_states = []
        for state_identifier, pos in zip(possible_states, possible_positions):
            next_states.append((self.estimated_values[state_identifier], pos))

        # 다음 상태 중 무작위로 행동을 선택하기 위해 shuffle 호출
        np.random.shuffle(next_states)

        # 가장 기댓값이 높은 행동이 앞에 오도록 정렬
        next_states.sort(key=lambda x: x[0], reverse=True)
        i, j = next_states[0][1]  # 가장 기댓값이 높은 행동 선택
        return i, j

    # 현재 정책 파일로 저장
    def save_policy(self):
        with open('policy_{0}.bin'.format("first" if self.player_int == PLAYER_1 else "seconds"), 'wb') as f:
            pickle.dump(self.estimated_values, f)

    # 저장된 파일에서 정책 불러오기
    def load_policy(self):
        with open('policy_{0}.bin'.format("first" if self.player_int == PLAYER_1 else "seconds"), 'rb') as f:
            self.estimated_values = pickle.load(f)


################################################################
# 인간 플레이어용 클래스
# 아래 알파벳을 입력하여 해당 칸에 o/* 표시
# | q | w | e |
# | a | s | d |
# | z | x | c |
class Human_Player:
    def __init__(self, player_int):
        self.player_int = player_int

        self.keys = ['q', 'w', 'e', 'a', 's', 'd', 'z', 'x', 'c']
        self.state = None
        self.env = TicTacToe()

        generate_all_states(
            state=INITIAL_STATE,
            player_int=player_int
        )

    def reset(self):
        self.append_state(INITIAL_STATE)

    def append_state(self, state):
        self.state = state

    # 사용자에게 입력을 받아서 행동
    def act(self):
        key = input("표시할 위치를 입력하십시오:")
        while True:
            if key in self.keys:
                data = self.keys.index(key)
                i = data // BOARD_COLS
                j = data % BOARD_COLS
                return i, j

            elif key == 'exit':
                return -1, -1

            else:
                key = input("다시 입력해주세요:")


# 훈련
def train(max_episodes, print_every_n=500):
    epsilon = 0.01  # 탐욕적 방법을 따르지 않고 무작위로 행동할 확률
    env = TicTacToe()

    player_1 = Player(PLAYER_1, epsilon=epsilon) # 선공 플레이어. 정수값 1로 표현
    player_2 = Player(PLAYER_2, epsilon=epsilon) # 후공 플레이어. 정수값 -1로 표현

    player_1.initialize_estimated_values()
    player_2.initialize_estimated_values()
    print("Player 1 - 내부 사용 정수값: {0}".format(player_1.player_int))
    print("Player 2 - 내부 사용 정수값: {0}".format(player_2.player_int))

    # 각 플레이어의 승리 횟수
    num_player_1_wins, num_player_2_wins, num_ties = 0.0, 0.0, 0.0

    episode_list = [0]
    tie_rate_list = [0.0]
    player1_win_rate_list = [0.0]
    player2_win_rate_list = [0.0]

    for episode in range(1, max_episodes + 1):
        env.reset()
        player_1.reset()
        player_2.reset()

        current_player = player_1
        winner = None
        done = False

        while not done:
            # 현재 상태와 정책에 의한 플레이어의 행동
            i, j = current_player.act()
            next_state, _, done, info = env.step(action=(i, j))
            #print(next_state, _, done, info)

            player_1.append_state(next_state)
            player_2.append_state(next_state)

            if done:
                winner = info['winner']
            else:
                if current_player == player_1:
                    current_player = player_2
                else:
                    current_player = player_1

        if winner == PLAYER_1:
            num_player_1_wins += 1
        elif winner == PLAYER_2:
            num_player_2_wins += 1
        elif winner == 0:
            num_ties += 1

        # 게임 종료 후 가치 함수 갱신
        player_1.update_estimated_values()
        player_2.update_estimated_values()

        # print_every_n 번째 게임마다 현재 결과 콘솔에 출력
        if episode % print_every_n == 0:
            print('에피소드: {0}, 비기는 비율: {1:.02f}, 플레이어 1 승률: {2:.02f}, 플레이어 2 승률: {3:.02f}'.format(
                episode,
                num_ties / episode,
                num_player_1_wins / episode,
                num_player_2_wins / episode
            ))
            episode_list.append(episode)
            tie_rate_list.append(num_ties / episode)
            player1_win_rate_list.append(num_player_1_wins / episode)
            player2_win_rate_list.append(num_player_2_wins / episode)


    # 학습 종료 후 정책 저장
    player_1.save_policy()
    player_2.save_policy()

    draw_figure_after_train(episode_list, tie_rate_list, player1_win_rate_list, player2_win_rate_list)


# 학습이 끝난 정책으로 에이전트끼리 경쟁
def self_play(turns):
    epsilon = 0.0  # epsilon = 0이므로 학습된 정책대로만 행동
    env = TicTacToe()

    player_1 = Player(PLAYER_1, epsilon=epsilon) # 선공 플레이어. 정수값 1로 표현
    player_2 = Player(PLAYER_2, epsilon=epsilon) # 후공 플레이어. 정수값 -1로 표현

    player_1.load_policy()  # 저장된 정책 불러오기
    player_2.load_policy()

    num_player_1_wins = 0.0
    num_player_2_wins = 0.0
    num_ties = 0.0

    # 게임 진행
    for _ in range(turns):
        env.reset()
        player_1.reset()
        player_2.reset()

        current_player = player_1
        winner = None
        done = False

        while not done:
            # 현재 상태와 정책에 의한 플레이어의 행동
            i, j = current_player.act()
            next_state, _, done, info = env.step(action=(i, j))
            player_1.append_state(next_state)
            player_2.append_state(next_state)

            if done:
                winner = info['winner']
            else:
                if current_player == player_1:
                    current_player = player_2
                else:
                    current_player = player_1

        if winner == PLAYER_1:
            num_player_1_wins += 1
        elif winner == PLAYER_2:
            num_player_2_wins += 1
        elif winner == 0:
            num_ties += 1

    # 학습이 잘 이루어진 경우, 항상 무승부로 종료
    print('\n총 {0}회 시행, 비기는 비율: {1:.02f}, 플레이어 1 승률: {2:.02f}, 플레이어 2 승률: {3:.02f}\n'.format(
        turns,
        num_ties / turns,
        num_player_1_wins / turns,
        num_player_2_wins / turns
    ))


# tic-tac-toe 게임은 제로섬 게임
# 만약 양 플레이어가 최적의 전략으로 게임에 임한다면, 모든 게임은 무승부
# 그러므로 인공지능이 후공일 때 최소한 무승부가 되는지 확인
def play_with_human():
    epsilon = 0.0
    env = TicTacToe()
    env.reset()

    player_1 = Human_Player(PLAYER_1)
    player_2 = Player(PLAYER_2, epsilon=epsilon)
    player_1.reset()
    player_2.reset()
    player_2.load_policy()

    while True:
        env.reset()
        player_1.reset()
        player_2.reset()
        player_2.load_policy()

        current_player = player_1
        done = False
        winner = None

        # 게임 진행
        while not done:
            # 현재 상태와 정책에 의한 플레이어의 행동
            env.render()
            i, j = current_player.act()
            next_state, _, done, info = env.step(action=(i, j))

            player_1.append_state(next_state)
            player_2.append_state(next_state)

            if done:
                winner = info['winner']
            else:
                if current_player == player_1:
                    current_player = player_2
                else:
                    current_player = player_1

        if winner == PLAYER_1:
            print("You win!\n")
        elif winner == PLAYER_2:
            print("You lose!\n")
        elif winner == 0:
            print("It is a tie!\n")
        else:
            break


# 훈련 중 반복 횟수 별 비긴 비율, 플레이어 1 승리 비율, 플레이어 2 승리 비율 비교 그림 출력
def draw_figure_after_train(episode_list, tie_rate_list, player1_win_rate_list, player2_win_rate_list):
    plt.figure()
    plt.plot(episode_list, tie_rate_list, label='비긴 비율', linestyle='-')
    plt.plot(episode_list, player1_win_rate_list, label='플레이어 1의 승리 비율', linestyle='--')
    plt.plot(episode_list, player2_win_rate_list, label='플레이어 2의 승리 비율', linestyle=':')
    plt.xlabel('훈련 반복 횟수')
    plt.ylabel('비율')
    plt.legend()
    plt.savefig('images/tic_tac_toe_gym_train.png')
    plt.show()
    plt.close()


if __name__ == '__main__':
    print("총 상태 개수: {0}".format(len(ALL_STATES)))
    train(max_episodes=50000)
    self_play(turns=1000)
    play_with_human()
