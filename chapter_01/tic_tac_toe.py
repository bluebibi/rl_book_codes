#######################################################################
# Copyright (C)                                                       #
# 2016 - 2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)           #
# 2016 Jan Hakenberg(jan.hakenberg@gmail.com)                         #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import pickle

BOARD_ROWS = 3
BOARD_COLS = 3
BOARD_SIZE = BOARD_ROWS * BOARD_COLS

################################################################
# 게임판 상태의 저장, 출력 그리고 종료 판정을 수행하는 State 클래스
class State:
    def __init__(self):
        # 게임판은 n * n 크기의 배열로 표현
        # 게임판에서 플레이어는 정수값으로 구분
        # 1 : 선공 플레이어, -1 : 후공 플레이어, 0 : 초기 공백 상태
        self.data = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.winner = None
        self.hash_val = None # 게임의 각 상태들을 구분짓기 위한 해시값
        self.end = None

    # 특정 상태에서의 유일한 해시값 계산
    def hash(self):
        if self.hash_val is None:
            self.hash_val = 0
            for i in np.nditer(self.data):  # 게임판 각 칸을 순회하며 해시값 갱신
                self.hash_val = self.hash_val * 3 + i + 1
        return self.hash_val

    # 플레이어가 게임을 이겼는지, 비겼는지 판단
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
                self.winner = result / 3
                self.end = True
                return self.end

        # 무승부 확인
        sum_values = np.sum(np.abs(self.data))
        if sum_values == BOARD_SIZE:
            self.winner = 0
            self.end = True
            return self.end

        # 게임이 미종료
        self.end = False
        return self.end

    # 게임판의 다음 상태(t + 1) 반환
    def next_state(self, i, j, player_symbol):
        new_state = State()
        new_state.data = np.copy(self.data)     # 현재 상태(t) 복사
        new_state.data[i, j] = player_symbol    # 플레이어의 행동(i,j 위치에 표시) 반영
        return new_state

    # 게임판 출력
    def print_board(self):
        for i in range(BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(BOARD_COLS):
                out += 'x0*'[int(self.data[i, j]) + 1] + ' | '
            print(out)
        print('-------------')


# 발생 가능한 모든 게임 상태 집합 생성
def generate_all_states(current_state, current_player_symbol, all_states):
    for i in range(BOARD_ROWS):
        for j in range(BOARD_COLS):
            if current_state.data[i][j] == 0:
                # 도달 가능한 다음 상태 생성
                new_state = current_state.next_state(i, j, current_player_symbol)
                new_hash = new_state.hash()
                if new_hash not in all_states:
                    is_end = new_state.is_end()
                    all_states[new_hash] = (new_state, is_end)  # 모든 게임 상태 집합 갱신
                    if not is_end:                              # 게임 미종료시 재귀 호출로 다음 상태 이동
                        generate_all_states(new_state, -current_player_symbol, all_states)


# 발생 가능한 모든 게임 상태 집합 반환
def get_all_states():
    current_player_symbol = 1
    current_state = State()

    # 맨 처음 상태 초기화
    all_states = dict()
    all_states[current_state.hash()] = (current_state, current_state.is_end())
    # 발생 가능한 모든 상태 추가
    generate_all_states(current_state, current_player_symbol, all_states)
    return all_states


# 발생 가능한 모든 게임판 상태의 집합
all_states = get_all_states()

################################################################
# 플레이어 1,2 간의 게임 진행을 담당하는 Game 클래스
class Game:
    def __init__(self, player1, player2):
        self.p1 = player1   # 선공 플레이어. 정수값 1로 표현
        self.p2 = player2   # 후공 플레이어. 정수값 -1로 표현
        self.current_player = None
        self.p1_player_symbol = 1
        self.p2_player_symbol = -1
        self.p1.set_player_symbol(self.p1_player_symbol)
        self.p2.set_player_symbol(self.p2_player_symbol)
        self.current_state = State()

    def reset(self):
        self.p1.reset()
        self.p2.reset()

    # 턴제 게임 구현을 위한 함수
    # 호출 시마다 플레이어1, 2를 번갈아가며 반환
    def alternate(self):
        while True:
            yield self.p1
            yield self.p2

    def play(self, print_board=False):
        alternator = self.alternate()   # 턴마다 행동할 플레이어를 반환하는 generator
        
        self.reset()                    # 게임 상태 초기화
        current_state = State()
        self.p1.set_state(current_state)
        self.p2.set_state(current_state)
        if print_board:
            current_state.print_board()
            
        # 게임 진행
        while True:
            player = next(alternator)

            # 현재 상태와 정책에 의한 플레이어의 행동
            i, j, player_symbol = player.act()
            # 플레이어의 행동에 의한 다음 상태 갱신
            next_state_hash = current_state.next_state(i, j, player_symbol).hash()
            current_state, is_end = all_states[next_state_hash]
            self.p1.set_state(current_state)
            self.p2.set_state(current_state)

            if print_board:
                current_state.print_board()
            if is_end:
                return current_state.winner

################################################################
# tic tac toe 환경과 상호작용하는 플레이어 에이전트
class Player:
    def __init__(self, learning_rate=0.1, epsilon=0.1):
        # 특정 상태에서의 가치 함수들의 집합
        self.estimated_values = dict()
        # 가치 함수 갱신 비율
        self.learning_rate = learning_rate
        # 탐욕적 방법으로 행동하지 않을 확률
        self.epsilon = epsilon
        self.states = []
        self.greedy = []
        self.player_symbol = 0

    def reset(self):
        self.states = []
        self.greedy = []

    def set_state(self, state):
        self.states.append(state)
        self.greedy.append(True)

    def set_player_symbol(self, player_symbol):
        # Player 선공, 후공 구분을 위한 symbol 갱신
        self.player_symbol = player_symbol

        # 가치 함수 초기화
        for hash_val in all_states:
            state, is_end = all_states[hash_val]
            if is_end:
                if state.winner == self.player_symbol:
                    self.estimated_values[hash_val] = 1.0
                elif state.winner == 0:
                    self.estimated_values[hash_val] = 0.5
                else:
                    self.estimated_values[hash_val] = 0
            else:
                self.estimated_values[hash_val] = 0.5

    # 게임 1회 종료 후 가치 함수 갱신
    def update_estimated_values(self):
        states = [state.hash() for state in self.states]

        for i in reversed(range(len(states) - 1)):      # 게임 처음 상태부터 마지막 상태까지의 역순으로
            state = states[i]
            temporal_difference = self.greedy[i] * (    # 행동 이후 상태와 이전 상태 기댓값의 차이만큼
                self.estimated_values[states[i + 1]] - self.estimated_values[state]
            )
            # 해당 상태의 가치 함수를 갱신
            # 공식 V(S(t)) <- V(S(t)) + a[V(S(t+1)) - V(S(t))], a = learning_rate
            self.estimated_values[state] += self.learning_rate * temporal_difference

    # 현재 상태에 기반하여 행동 결정
    def act(self):
        state = self.states[-1]
        
        # 현재 상태에서 도달 가능한 다음 상태들을 저장
        possible_states = []
        possible_positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if state.data[i, j] == 0:
                    possible_positions.append([i, j])
                    possible_states.append(state.next_state(
                        i, j, self.player_symbol).hash())

        # epsilon 값에 의해 확률적으로 임의의 행동 수행
        if np.random.rand() < self.epsilon:
            action = possible_positions[np.random.randint(len(possible_positions))]
            action.append(self.player_symbol)
            self.greedy[-1] = False
            return action

        next_states = []
        for hash_val, pos in zip(possible_states, possible_positions):
            next_states.append((self.estimated_values[hash_val], pos))

        # python의 sort는 stable하므로 기댓값이 동일한 다음 상태 중 무작위로 행동을 선택하기 위해 shuffle 호출
        np.random.shuffle(next_states)

        # 가장 기댓값이 높은 행동이 앞에 오도록 정렬
        next_states.sort(key=lambda x: x[0], reverse=True)
        action = next_states[0][1]               # 가장 기댓값이 높은 행동 선택
        action.append(self.player_symbol)
        return action

    # 현재 정책 파일로 저장
    def save_policy(self):
        with open('policy_%s.bin' % ('first' if self.player_symbol == 1 else 'second'), 'wb') as f:
            pickle.dump(self.estimated_values, f)

    # 저장된 파일에서 정책 불러오기
    def load_policy(self):
        with open('policy_%s.bin' % ('first' if self.player_symbol == 1 else 'second'), 'rb') as f:
            self.estimated_values = pickle.load(f)


################################################################
# 인간 플레이어용 클래스
# 아래 알파벳을 입력하여 해당 칸에 o/* 표시
# | q | w | e |
# | a | s | d |
# | z | x | c |
class HumanPlayer:
    def __init__(self, **kwargs):
        self.keys = ['q', 'w', 'e', 'a', 's', 'd', 'z', 'x', 'c']
        self.player_symbol = None
        self.state = None

    def reset(self):
        pass

    def set_state(self, state):
        self.state = state

    def set_player_symbol(self, player_symbol):
        self.player_symbol = player_symbol

    # 사용자에게 입력을 받아서 행동
    def act(self):
        self.state.print_board()
        key = input("표시할 위치를 입력하십시오:")
        while True:
            if key in self.keys:
                data = self.keys.index(key)
                i = data // BOARD_COLS
                j = data % BOARD_COLS
                return i, j, self.player_symbol
            elif key == 'exit':
                exit(0)
            else:
                key = input("다시 입력해주세요:")


def train(epochs, print_every_n=500):
    epsilon = 0.01  # 탐욕적 방법을 따르지 않고 무작위로 행동할 확률
    player1 = Player(epsilon=epsilon)
    player2 = Player(epsilon=epsilon)
    game = Game(player1, player2)
    # 각 플레이어의 승리 횟수
    num_player1_win, num_player2_win = 0.0, 0.0
    for i in range(1, epochs + 1):
        winner = game.play(print_board=False)
        if winner == 1:
            num_player1_win += 1
        if winner == -1:
            num_player2_win += 1
        # print_every_n 번째 게임마다 현재 결과 콘솔에 출력
        if i % print_every_n == 0:
            print('Epoch %d, 플레이어 1 승률: %.02f, 플레이어 2 승률: %.02f' % (i, num_player1_win / i, num_player2_win / i))

        # 게임 종료 후 가치 함수 갱신
        player1.update_estimated_values()
        player2.update_estimated_values()
        game.reset()
        
    # 학습 종료 후 정책 저장
    player1.save_policy()
    player2.save_policy()


# 학습이 끝난 정책으로 인공지능 플레이어끼리 경쟁
def compete(turns):
    player1 = Player(epsilon=0) # epsilon = 0이므로 학습된 정책대로만 행동
    player2 = Player(epsilon=0)
    game = Game(player1, player2)
    player1.load_policy()       # 저장된 정책 불러오기
    player2.load_policy()
    num_player1_win = 0.0
    num_player2_win = 0.0
    
    # 게임 진행
    for _ in range(turns):
        winner = game.play()
        if winner == 1:
            num_player1_win += 1
        if winner == -1:
            num_player2_win += 1
        game.reset()
        
    # 학습이 잘 이루어진 경우, 항상 무승부로 종료
    print('총 %d회 시행, 플레이어 1 승률 %.02f, 플레이어 2 승률 %.02f' % (turns, num_player1_win / turns, num_player2_win / turns))


# tic tac toe 게임은 제로섬 게임
# 만약 양 플레이어가 최적의 전략으로 게임에 임한다면, 모든 게임은 무승부
# 그러므로 인공지능이 후공일 때 최소한 무승부가 보장되는지 확인
def play():
    while True:
        player1 = HumanPlayer()     # 인간 플레이어
        player2 = Player(epsilon=0) # 인공지능 플레이어
        game = Game(player1, player2)
        player2.load_policy()       # 학습된 정책 불러오기
        winner = game.play()
        if winner == player2.player_symbol:
            print("패배하였습니다!")
        elif winner == player1.player_symbol:
            print("승리하였습니다!")
        else:
            print("비겼습니다!")


################################################################
## MAIN ##
if __name__ == '__main__':
    train(int(1e5))
    compete(int(1e3))
    play()