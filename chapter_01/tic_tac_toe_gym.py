import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import gym

# 이미지 저장 경로 확인 및 생성
if not os.path.exists('images/'):
    os.makedirs('images/')

plt.rcParams["font.family"] = 'NanumBarunGothic'
plt.rcParams["font.size"] = 12
mpl.rcParams['axes.unicode_minus'] = False

BOARD_ROWS = 3
BOARD_COLS = 3
BOARD_SIZE = BOARD_ROWS * BOARD_COLS
PLAYER_INT_TO_SYMBOL = ['*', 'O', 'X']

PLAYER_1_INT = 1
PLAYER_2_INT = -1


#########################################################
# 게임판 상태의 저장, 출력 그리고 종료 판정을 수행하는 State 클래스   #
#########################################################
class State:
    def __init__(self):
        # 게임판은 n * n 크기의 배열로 표현
        # 게임판에서 플레이어는 정수값으로 구분
        # 1 : 선공 플레이어, -1 : 후공 플레이어, 0 : 초기 공백 상태
        self.data = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.winner = None
        self.hash_val = None  # 게임의 각 상태들을 구분짓기 위한 해시값
        self.end = None

    # 특정 상태에서의 유일한 해시값 계산
    def hash(self):
        if self.hash_val is None:
            self.hash_val = 0
            it = np.nditer(self.data)
            for i in range(BOARD_ROWS):
                for j in range(BOARD_COLS):
                    self.hash_val = self.hash_val * 3 + self.data[i, j] + 1
        return self.hash_val

    # 플레이어가 종료 상태에 있는지 판단.
    # 플레이어가 게임을 이기거나, 지거나, 비겼다면 True 반환, 그 외는 False 반환
    def is_end(self):
        if self.end is not None:
            return self.end

        results = []
        for i in range(BOARD_ROWS):
            results.append(np.sum(self.data[i, :]))

        for i in range(BOARD_COLS):
            results.append(np.sum(self.data[:, i]))

        # 게임판 대각선 승리조건 확인
        trace = 0
        reverse_trace = 0
        for i in range(BOARD_ROWS):
            trace += self.data[i, i]
            reverse_trace += self.data[i, BOARD_ROWS - 1 - i]
        results.append(trace)
        results.append(reverse_trace)

        for result in results:
            if result == 3 or result == -3:
                self.end = True
                if result == 3:
                    self.winner = PLAYER_1_INT
                else:
                    self.winner = PLAYER_2_INT
                return self.end

        # 무승부 확인
        sum_values = np.sum(np.abs(self.data))
        if sum_values == BOARD_SIZE:
            self.winner = 0
            self.end = True
            return self.end

        # 게임이 아직 종료되지 않음
        self.end = False
        return self.end

    # 게임판 출력
    def print_board(self):
        for i in range(BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(BOARD_COLS):
                out += PLAYER_INT_TO_SYMBOL[int(self.data[i, j])] + ' | '
            print(out)
        print('-------------')

    def __repr__(self):
        self_str = ""
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self_str += PLAYER_INT_TO_SYMBOL[int(self.data[i, j])]
            if i < BOARD_ROWS - 1:
                self_str += ','
        return self_str


class Board:
    def __init__(self):
        # 초기 상태 초기화
        self.initial_player_int = PLAYER_1_INT
        self.initial_state = State()

        # 발생 가능한 모든 게임 상태 집합
        self.all_states = dict()
        self.all_states[self.initial_state.hash()] = (self.initial_state, self.initial_state.is_end())
        self.generate_all_states(state=self.initial_state, player_int=self.initial_player_int)

    def get_new_state(self, i, j, state_data, player_int):
        new_state = State()
        # 주어진 상태의 게임판 상황 복사
        new_state.data = np.copy(state_data)
        # 플레이어의 행동(i, j 위치에 표시) 반영
        new_state.data[i, j] = player_int
        return new_state

    # 주어진 상태 및 현재 플레이어 심볼에 대하여 발생 가능한 모든 게임 상태 집합 생성
    def generate_all_states(self, state, player_int):
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if state.data[i][j] == 0:
                    # 도달 가능한 새로운 상태 생성
                    new_state = self.get_new_state(i, j, state.data, player_int)

                    # 새로운 상태의 해시값 가져오기
                    new_hash = new_state.hash()

                    if new_hash not in self.all_states:
                        # 모든 게임 상태 집합 갱신
                        self.all_states[new_hash] = (new_state, new_state.is_end())
                        # 게임 미종료시 재귀 호출로 새로운 상태 계속 생성
                        if not new_state.is_end():
                            self.generate_all_states(new_state, -player_int)


class Player:
    def __init__(self, board, step_size=0.1, epsilon=0.1):
        # 게임 보드
        self.board = board

        # 특정 상태에서의 가치 함수들의 집합
        self.estimated_values = dict()

        # 가치 함수 갱신 비율
        self.step_size = step_size

        # 탐욕적 방법으로 행동하지 않을 확률
        self.epsilon = epsilon
        self.visited_states = []
        self.player_int = 0

    def reset(self, player_int):
        # Player_1(선공), Player_2(후공) 구분을 위한 값 갱신
        self.player_int = player_int
        self.visited_states.clear()

    def append_state(self, state):
        self.visited_states.append(state)

    def initialize_estimated_values(self):
        # 가치 함수 초기화
        for hash_val in self.board.all_states:
            state, is_end = self.board.all_states[hash_val]
            if is_end:
                if state.winner == self.player_int:
                    self.estimated_values[hash_val] = 1.0
                elif state.winner == 0:
                    self.estimated_values[hash_val] = 0.5
                else:
                    self.estimated_values[hash_val] = 0
            else:
                self.estimated_values[hash_val] = 0.5

    # 게임 1회 종료 후 가치 함수 갱신
    def update_estimated_values(self):
        states_hash_values = [state.hash() for state in self.visited_states]

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
                    new_state_hash = self.board.get_new_state(i, j, state.data, self.player_int).hash()
                    assert new_state_hash in self.board.all_states
                    possible_states.append(new_state_hash)

        # epsilon 값에 의해 확률적으로 임의의 행동 수행
        if np.random.rand() < self.epsilon:
            i, j = possible_positions[np.random.randint(len(possible_positions))]
            return i, j, self.player_int

        next_states = []
        for hash_val, pos in zip(possible_states, possible_positions):
            next_states.append((self.estimated_values[hash_val], pos))

        # 다음 상태 중 무작위로 행동을 선택하기 위해 shuffle 호출
        np.random.shuffle(next_states)

        # 가장 기댓값이 높은 행동이 앞에 오도록 정렬
        next_states.sort(key=lambda x: x[0], reverse=True)
        i, j = next_states[0][1]  # 가장 기댓값이 높은 행동 선택
        return i, j, self.player_int

    # 현재 정책 파일로 저장
    def save_policy(self):
        with open('policy_%s.bin' % ('first' if self.player_int == PLAYER_1_INT else 'second'), 'wb') as f:
            pickle.dump(self.estimated_values, f)

    # 저장된 파일에서 정책 불러오기
    def load_policy(self):
        with open('policy_%s.bin' % ('first' if self.player_int == PLAYER_1_INT else 'second'), 'rb') as f:
            self.estimated_values = pickle.load(f)

################################################################
# 인간 플레이어용 클래스
# 아래 알파벳을 입력하여 해당 칸에 o/* 표시
# | q | w | e |
# | a | s | d |
# | z | x | c |
class Human_Player:
    def __init__(self, **kwargs):
        self.keys = ['q', 'w', 'e', 'a', 's', 'd', 'z', 'x', 'c']
        self.player_int = None
        self.state = None

    def reset(self, player_int):
        self.player_int = player_int

    def append_state(self, state):
        self.state = state

    # 사용자에게 입력을 받아서 행동
    def act(self):
        self.state.print_board()
        key = input("표시할 위치를 입력하십시오:")
        while True:
            if key in self.keys:
                data = self.keys.index(key)
                i = data // BOARD_COLS
                j = data % BOARD_COLS
                return i, j, self.player_int

            elif key == 'exit':
                return -1, -1, None

            else:
                key = input("다시 입력해주세요:")


################################################################
# 플레이어 1,2 간의 게임 진행을 담당하는 Env 클래스
class TIC_TAC_TOE_ENV(gym.Env):
    def __init__(self, player1, player2, ):
        self.board = Board()
        self.p1 = player1  # 선공 플레이어. 정수값 1로 표현
        self.p2 = player2  # 후공 플레이어. 정수값 -1로 표현

    def reset(self):
        self.current_player = None
        self.p1.reset(PLAYER_1_INT)
        self.p2.reset(PLAYER_2_INT)
        # 턴마다 행동할 플레이어를 반환하는 generator
        self.alternator = self.alternate()
        self.current_state = self.board.initial_state

        self.p1.append_state(self.board.initial_state)
        self.p2.append_state(self.board.initial_state)

    # 게임 진행을 위해 턴마다 호출
    def step(self, action=None):
        player = next(self.alternator)

        # 현재 상태와 정책에 의한 플레이어의 행동
        i, j, player_int = player.act()

        if player_int is None:
            return None

        # 플레이어의 행동에 의한 다음 상태 갱신
        next_state_hash = self.board.get_new_state(i, j, self.current_state.data, player_int).hash()
        assert next_state_hash in self.board.all_states

        next_state, is_end = self.board.all_states[next_state_hash]
        self.p1.append_state(next_state)
        self.p2.append_state(next_state)

        if is_end:
            return next_state.winner
        else:
            self.current_state = next_state
            return 2

    def render(self, mode='human'):
        self.current_state.print_board()

    # 턴제 게임 구현을 위한 함수
    # 호출 시마다 플레이어1, 2를 번갈아가며 반환
    def alternate(self):
        while True:
            yield self.p1
            yield self.p2



# 훈련
def train(epochs, print_every_n=500):
    epsilon = 0.01  # 탐욕적 방법을 따르지 않고 무작위로 행동할 확률

    player1 = Player(Board(), epsilon=epsilon)
    player2 = Player(Board(), epsilon=epsilon)

    env = TIC_TAC_TOE_ENV(player1, player2)
    env.reset()
    player1.initialize_estimated_values()
    player2.initialize_estimated_values()
    print("총 상태 개수: {0}".format(len(env.board.all_states)))
    print("Player 1 - 내부 사용 정수값: {0}".format(player1.player_int))
    print("Player 2 - 내부 사용 정수값: {0}".format(player2.player_int))

    # 각 플레이어의 승리 횟수
    num_player1_wins, num_player2_wins, num_ties = 0.0, 0.0, 0.0

    episode_list = [0]
    tie_rate_list = [0.0]
    player1_win_rate_list = [0.0]
    player2_win_rate_list = [0.0]

    for i in range(1, epochs + 1):
        winner = 2  # 진행 중인 게임은 2로 표시
        while winner == 2:
            winner = env.step()

        if winner == PLAYER_1_INT:
            num_player1_wins += 1
        elif winner == PLAYER_2_INT:
            num_player2_wins += 1
        elif winner == 0:
            num_ties += 1

        # 게임 종료 후 가치 함수 갱신
        player1.update_estimated_values()
        player2.update_estimated_values()

        # print_every_n 번째 게임마다 현재 결과 콘솔에 출력
        if i % print_every_n == 0:
            print('Epoch {0}, 비기는 비율: {1:.02f}, 플레이어 1 승률: {2:.02f}, 플레이어 2 승률: {3:.02f}'.format(
                i,
                num_ties / i,
                num_player1_wins / i,
                num_player2_wins / i
            ))
            episode_list.append(i)
            tie_rate_list.append(num_ties / i)
            player1_win_rate_list.append(num_player1_wins / i)
            player2_win_rate_list.append(num_player2_wins / i)

        env.reset()

    # 학습 종료 후 정책 저장
    player1.save_policy()
    player2.save_policy()

    return episode_list, tie_rate_list, player1_win_rate_list, player2_win_rate_list

# 학습이 끝난 정책으로 에이전트끼리 경쟁
def self_play(turns):
    # epsilon = 0이므로 학습된 정책대로만 행동
    player1 = Player(Board(), epsilon=0)
    player2 = Player(Board(), epsilon=0)

    env = TIC_TAC_TOE_ENV(player1, player2)
    env.reset()

    player1.load_policy()  # 저장된 정책 불러오기
    player2.load_policy()

    num_player1_wins = 0.0
    num_player2_wins = 0.0
    num_ties = 0.0

    # 게임 진행
    for _ in range(turns):
        winner = 2
        while winner == 2:
            winner = env.step()

        if winner == PLAYER_1_INT:
            num_player1_wins += 1
        elif winner == PLAYER_2_INT:
            num_player2_wins += 1
        elif winner == 0:
            num_ties += 1

        env.reset()

    # 학습이 잘 이루어진 경우, 항상 무승부로 종료
    print('총 {0}회 시행, 비기는 비율: {1:.02f}, 플레이어 1 승률: {2:.02f}, 플레이어 2 승률: {3:.02f}'.format(
        turns,
        num_ties / turns,
        num_player1_wins / turns,
        num_player2_wins / turns
    ))


# tic-tac-toe 게임은 제로섬 게임
# 만약 양 플레이어가 최적의 전략으로 게임에 임한다면, 모든 게임은 무승부
# 그러므로 인공지능이 후공일 때 최소한 무승부가 되는지 확인
def play_with_human():
    while True:
        player1 = Human_Player()
        player2 = Player(Board(), epsilon=0)

        env = TIC_TAC_TOE_ENV(player1, player2)
        env.reset()

        player2.load_policy()

        winner = 2
        while winner == 2:
            winner = env.step()

        if winner == PLAYER_1_INT:
            print("You win!")
        elif winner == PLAYER_2_INT:
            print("You lose!")
        elif winner == 0:
            print("It is a tie!")
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
    episode_list, tie_rate_list, player1_win_rate_list, player2_win_rate_list = train(epochs = 100000)
    draw_figure_after_train(episode_list, tie_rate_list, player1_win_rate_list, player2_win_rate_list)
    self_play(turns=1000)
    play_with_human()
