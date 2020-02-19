# -------------------------------
# |(0,0)|(0,1)|(0,2)|(0,3)|(0,4)|
# |(1,0)|(1,1)|(1,2)|(1,3)|(1,4)|
# |(2,0)|(2,1)|(2,2)|(2,3)|(2,4)|
# |(3,0)|(3,1)|(3,2)|(3,3)|(3,4)|
# |(4,0)|(4,1)|(4,2)|(4,3)|(4,4)|
# -------------------------------

class GridWorld:
    def __init__(
            self,
            height=5,
            width=5,
            start_state=(0, 0),
            terminal_state=[(4, 4)],
            transition_reward=0.0,
            terminal_reward=1.0
    ):
        # 그리드월드의 세로 길이
        self.HEIGHT = height

        # 그리드월드의 가로 길이
        self.WIDTH = width

        self.num_states = self.WIDTH * self.HEIGHT

        self.STATES = []
        for i in range(self.HEIGHT):
            for j in range(self.WIDTH):
                self.STATES.append((i, j))
        for state in terminal_state:     # 터미널 스테이트 제거
            self.STATES.remove(state)

        # 모든 가능한 행동
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_LEFT = 2
        self.ACTION_RIGHT = 3
        self.ACTION_SYMBOLS = ["\u2191", "\u2193", "\u2190", "\u2192"]
        self.ACTIONS = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT]
        self.num_actions = len(self.ACTIONS)

        # 시작 상태 위치
        self.START_STATE = start_state

        # 종료 상태 위치
        self.GOAL_STATES = terminal_state

        # 최대 타임 스텝
        self.max_steps = float('inf')

        self.transition_reward = transition_reward

        self.terminal_reward = terminal_reward

        self.current_state = None

    def reset(self):
        self.current_state = self.START_STATE
        return self.current_state

    # take @action in @state
    # @return: (reward, new state)
    def step(self, state, action):
        x, y = state
        if action == self.ACTION_UP:
            x = max(x - 1, 0)
        elif action == self.ACTION_DOWN:
            x = min(x + 1, self.HEIGHT - 1)
        elif action == self.ACTION_LEFT:
            y = max(y - 1, 0)
        elif action == self.ACTION_RIGHT:
            y = min(y + 1, self.WIDTH - 1)

        if (x, y) in self.GOAL_STATES:
            reward = self.terminal_reward
        else:
            reward = self.transition_reward

        self.current_state = (x, y)
        return reward, (x, y)
