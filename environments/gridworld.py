import time

import gym

# -------------------------------
# |(0,0)|(0,1)|(0,2)|(0,3)|(0,4)|
# |(1,0)|(1,1)|(1,2)|(1,3)|(1,4)|
# |(2,0)|(2,1)|(2,2)|(2,3)|(2,4)|
# |(3,0)|(3,1)|(3,2)|(3,3)|(3,4)|
# |(4,0)|(4,1)|(4,2)|(4,3)|(4,4)|
# -------------------------------


class GridWorld(gym.Env):
    def __init__(
            self,
            height=5,
            width=5,
            start_state=(0, 0),
            terminal_state=[(4, 4)],
            transition_reward=0.0,
            terminal_reward=1.0,
            unique_steps=[]
    ):
        self.__version__ = "0.0.1"

        self.observation_space = gym.spaces.MultiDiscrete([height, width])
        self.action_space = gym.spaces.Discrete(4)

        # 그리드월드의 세로 길이
        self.HEIGHT = height

        # 그리드월드의 가로 길이
        self.WIDTH = width

        self.num_states = self.WIDTH * self.HEIGHT

        self.observation_space.STATES = []

        for i in range(self.HEIGHT):
            for j in range(self.WIDTH):
                self.observation_space.STATES.append((i, j))

        for state in terminal_state:     # 터미널 스테이트 제거
            self.observation_space.STATES.remove(state)

        # 모든 가능한 행동
        self.observation_space.ACTION_UP = 0
        self.observation_space.ACTION_DOWN = 1
        self.observation_space.ACTION_LEFT = 2
        self.observation_space.ACTION_RIGHT = 3
        self.observation_space.ACTION_SYMBOLS = ["\u2191", "\u2193", "\u2190", "\u2192"]
        self.observation_space.ACTIONS = [
            self.observation_space.ACTION_UP,
            self.observation_space.ACTION_DOWN,
            self.observation_space.ACTION_LEFT,
            self.observation_space.ACTION_RIGHT
        ]
        self.observation_space.num_actions = len(self.observation_space.ACTIONS)

        # 기본 GridWorld 에 추가되는 환경 조건들 집합
        self.unique_steps = unique_steps

        # 시작 상태 위치
        self.observation_space.START_STATE = start_state

        # 종료 상태 위치
        self.observation_space.GOAL_STATES = terminal_state

        # 최대 타임 스텝
        self.max_steps = float('inf')

        self.transition_reward = transition_reward

        self.terminal_reward = terminal_reward

        self.current_state = None

    def reset(self):
        self.current_state = self.observation_space.START_STATE
        return self.current_state

    # take @action in @state
    # @return: (reward, new state)
    def step(self, action):
        x, y = self.current_state

        # 기본 GridWorld에 추가된 조건들(ex. 함정, 웜홀 등) 적용
        # unique_step은 추가 조건 판정 및 수행에 관여하는 사용자 정의 함수
        # info['exec'] 로 추가 조건이 수행되었는지를 판정한다.
        for unique_step in self.unique_steps:
            (x, y), reward, done, info = unique_step((x, y), action)
            if info and info['exec']:
                return (x, y), reward, done, None

        if action == self.observation_space.ACTION_UP:
            x = max(x - 1, 0)
        elif action == self.observation_space.ACTION_DOWN:
            x = min(x + 1, self.HEIGHT - 1)
        elif action == self.observation_space.ACTION_LEFT:
            y = max(y - 1, 0)
        elif action == self.observation_space.ACTION_RIGHT:
            y = min(y + 1, self.WIDTH - 1)

        if (x, y) in self.observation_space.GOAL_STATES:
            reward = self.terminal_reward
        else:
            reward = self.transition_reward

        self.current_state = (x, y)

        if self.current_state in self.observation_space.GOAL_STATES:
            done = True
        else:
            done = False

        return (x, y), reward, done, None

    def render(self, mode='human'):
        for i in range(self.HEIGHT):
            print("-------------------------------")

            for j in range(self.WIDTH):
                if self.current_state[0] == i and self.current_state[1] == j:
                    print("|  {0}  ".format("*"), end="")
                else:
                    print("|     ".format("*"), end="")
            print("|")

            for j in range(self.WIDTH):
                print("|({0},{1})".format(i, j), end="")
            print("|")

        print("-------------------------------\n")


if __name__ == "__main__":
    env = GridWorld()
    env.reset()
    while True:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        env.render()
        time.sleep(1)
