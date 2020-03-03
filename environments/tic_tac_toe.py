import gym
import numpy as np

PLAYER_TO_SYMBOL = ['*', 'O', 'X']
PLAYER_1 = 1
PLAYER_2 = -1


#########################################################
# 게임판 상태의 저장, 출력 그리고 종료 판정을 수행하는 State 클래스   #
#########################################################
class State:
    def __init__(self, board_rows=3, board_cols=3):
        # 게임판은 n * n 크기의 배열로 표현
        # 게임판에서 플레이어는 정수값으로 구분
        # 1 : 선공 플레이어, -1 : 후공 플레이어, 0 : 초기 공백 상태
        self.data = np.zeros((board_rows, board_cols))
        self.winner = None
        self.hash_val = None  # 게임의 각 상태들을 구분짓기 위한 해시값
        self.end = None
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.board_size = board_rows * board_cols

    # 특정 상태에서의 유일한 해시값 계산
    def hash(self):
        if self.hash_val is None:
            self.hash_val = 0
            it = np.nditer(self.data)
            for i in range(self.board_rows):
                for j in range(self.board_cols):
                    self.hash_val = self.hash_val * 3 + self.data[i, j] + 1

        return self.hash_val

    # 플레이어가 종료 상태에 있는지 판단.
    # 플레이어가 게임을 이기거나, 지거나, 비겼다면 True 반환, 그 외는 False 반환
    def is_end(self):
        if self.end is not None:
            return self.end

        results = []
        for i in range(self.board_rows):
            results.append(np.sum(self.data[i, :]))

        for i in range(self.board_cols):
            results.append(np.sum(self.data[:, i]))

        # 게임판 대각선 승리조건 확인
        trace = 0
        reverse_trace = 0
        for i in range(self.board_rows):
            trace += self.data[i, i]
            reverse_trace += self.data[i, self.board_rows - 1 - i]
        results.append(trace)
        results.append(reverse_trace)

        for result in results:
            if result == 3 or result == -3:
                self.end = True
                if result == 3:
                    self.winner = PLAYER_1
                else:
                    self.winner = PLAYER_2
                return self.end

        # 무승부 확인
        sum_values = np.sum(np.abs(self.data))
        if sum_values == self.board_size:
            self.winner = 0
            self.end = True
            return self.end

        # 게임이 아직 종료되지 않음
        self.end = False
        return self.end

    # 게임판 출력
    def print_board(self):
        for i in range(self.board_rows):
            print('-------------')
            out = '| '
            for j in range(self.board_cols):
                out += PLAYER_TO_SYMBOL[int(self.data[i, j])] + ' | '
            print(out)
        print('-------------')

    def __repr__(self):
        self_str = ""
        for i in range(self.board_rows):
            for j in range(self.board_cols):
                self_str += PLAYER_TO_SYMBOL[int(self.data[i, j])]
            if i < self.board_rows - 1:
                self_str += ','
        return self_str


class Board:
    def __init__(self, board_rows, board_cols):
        # 초기 상태 초기화
        self.initial_player_int = PLAYER_1
        self.initial_state = State()
        self.board_rows = board_rows
        self.board_cols = board_cols

        # 발생 가능한 모든 게임 상태 집합
        self.all_states = dict()
        self.all_states[self.initial_state.hash()] = (self.initial_state, self.initial_state.is_end())

    def get_new_state(self, i, j, state_data, player_int):
        new_state = State()
        # 주어진 상태의 게임판 상황 복사
        new_state.data = np.copy(state_data)
        # 플레이어의 행동(i, j 위치에 표시) 반영
        new_state.data[i, j] = player_int
        return new_state

    # 주어진 상태 및 현재 플레이어 심볼에 대하여 발생 가능한 모든 게임 상태 집합 생성
    def generate_all_states(self, state, player_int):
        for i in range(self.board_rows):
            for j in range(self.board_cols):
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


################################################################
# 플레이어 1,2 간의 게임 진행을 담당하는 Env 클래스
class TicTacToe(gym.Env):
    def __init__(self, board_rows=3, board_cols=3):
        self.board = Board(board_rows, board_cols)
        self.BOARD_SIZE = board_rows * board_cols
        self.current_state = None
        self.current_player_int = None

    def reset(self):
        self.current_player_int = PLAYER_1
        self.current_state = self.board.initial_state

    # 게임 진행을 위해 턴마다 호출
    def step(self, action=None):
        # 플레이어의 행동에 의한 다음 상태 갱신
        next_state_hash = self.board.get_new_state(
            action[0],
            action[1],
            self.current_state.data,
            self.current_player_int
        ).hash()

        assert next_state_hash in self.board.all_states

        next_state, done = self.board.all_states[next_state_hash]

        if done:
            info = {'winner': next_state.winner}
        else:
            info = None

        self.current_state = next_state
        reward = 0.0

        if self.current_player_int == PLAYER_1:
            self.current_player_int = PLAYER_2
        else:
            self.current_player_int = PLAYER_1

        return next_state, reward, done, info

    def render(self, mode='human'):
        self.current_state.print_board()
