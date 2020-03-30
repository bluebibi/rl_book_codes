# 미로 내 장애물 및 시작 상태, 종료 상태 정보등을 모두 지닌 미로 클래스
class Maze:
    def __init__(self):
        # 미로의 가로 길이
        self.MAZE_WIDTH = 10

        # 미로의 세로 길이
        self.MAZE_HEIGHT = 6

        # 모든 가능한 행동
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_LEFT = 2
        self.ACTION_RIGHT = 3
        self.ACTIONS = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT]
        self.ACTION_SYMBOLS = ["\u2191", "\u2193", "\u2190", "\u2192"]

        # 시작 상태 위치
        self.START_STATE = (2, 1)

        # 종료 상태 위치
        self.TERMINAL_STATES = [(5, 9)]

        # 장애물들의 위치
        self.obstacles = [
            (0, 7), (0, 8),
            (1, 4), (1, 5),
            (2, 2), (2, 4), (2, 5), (2, 8),
            (3, 2), (3, 8),
            (4, 2), (4, 7), (4, 8),
            (5, 7), (5, 8)
        ]

        # Q 가치의 크기
        self.q_size = (self.MAZE_HEIGHT, self.MAZE_WIDTH, len(self.ACTIONS))

        # 최대 타임 스텝
        self.max_steps = float('inf')

        # track the resolution for this maze
        self.resolution = 1

    # take @action in @state
    # @return: [new state, reward]
    def step(self, state, action):
        x, y = state
        if action == self.ACTION_UP:
            x = max(x - 1, 0)
        elif action == self.ACTION_DOWN:
            x = min(x + 1, self.MAZE_HEIGHT - 1)
        elif action == self.ACTION_LEFT:
            y = max(y - 1, 0)
        elif action == self.ACTION_RIGHT:
            y = min(y + 1, self.MAZE_WIDTH - 1)

        if (x, y) in self.obstacles:
            x, y = state

        if (x, y) in self.TERMINAL_STATES:
            reward = 1.0
        else:
            reward = 0.0

        return reward, (x, y)


# 미로 내 장애물 및 시작 상태, 종료 상태 정보등을 모두 지닌 미로 클래스
class ChangingMaze(Maze):
    def __init__(self):
        super(ChangingMaze, self).__init__()
        self.original_obstacles = None
        self.new_obstacles = None

        # 장애물을 변경하는 타임 스텝
        self.obstacle_switch_time = None


    # extend a state to a higher resolution maze
    # @state: state in lower resoultion maze
    # @factor: extension factor, one state will become factor^2 states after extension
    def extend_state(self, state, factor):
        new_state = [state[0] * factor, state[1] * factor]
        new_states = []
        for i in range(0, factor):
            for j in range(0, factor):
                new_states.append([new_state[0] + i, new_state[1] + j])
        return new_states


    # extend a state into higher resolution
    # one state in original maze will become @factor^2 states in @return new maze
    def extend_maze(self, factor):
        new_maze = Maze()
        new_maze.MAZE_WIDTH = self.MAZE_WIDTH * factor
        new_maze.MAZE_HEIGHT = self.MAZE_HEIGHT * factor
        new_maze.START_STATE = [self.START_STATE[0] * factor, self.START_STATE[1] * factor]
        new_maze.TERMINAL_STATES = self.extend_state(self.TERMINAL_STATES[0], factor)
        new_maze.obstacles = []
        for state in self.obstacles:
            new_maze.obstacles.extend(self.extend_state(state, factor))
        new_maze.q_size = (new_maze.MAZE_HEIGHT, new_maze.MAZE_WIDTH, len(new_maze.actions))
        new_maze.resolution = factor
        return new_maze

