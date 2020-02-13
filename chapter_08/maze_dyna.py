import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.table import Table
from tqdm import tqdm
import heapq
from copy import deepcopy
import random

from chapter_08.maze import Maze

plt.rcParams["font.family"] = 'NanumBarunGothic'
plt.rcParams["font.size"] = 12
mpl.rcParams['axes.unicode_minus'] = False


class PriorityQueue:
    def __init__(self):
        self.priority_queue = []
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = 0

    def add_item(self, item, priority=0):
        if item in self.entry_finder:
            self.remove_item(item)
        entry = [priority, self.counter, item]
        self.counter += 1
        self.entry_finder[item] = entry
        heapq.heappush(self.priority_queue, entry)

    def remove_item(self, item):
        entry = self.entry_finder.pop(item)
        entry[-1] = self.REMOVED

    def pop_item(self):
        while self.priority_queue:
            priority, count, item = heapq.heappop(self.priority_queue)
            if item is not self.REMOVED:
                del self.entry_finder[item]
                return item, priority
        raise KeyError('pop from an empty priority queue')

    def is_empty(self):
        return not self.entry_finder


class DynaQParams:
    # 감가율
    gamma = 0.95

    # 탐색(exploration) 확률
    epsilon = 0.1

    # 스텝 사이즈
    alpha = 0.1

    # 경과 시간에 대한 가중치
    time_weight = 0

    # 계획에서의 수행 스텝 수
    planning_steps = 5

    # 총 수행 횟수 (성능에 대한 평균을 구하기 위함)
    runs = 10

    # 알고리즘 이름
    methods = ['Dyna-Q', 'Dyna-Q+']

    # 우선순위 코에 대한 임계값
    theta = 0


# choose an action based on epsilon-greedy algorithm
def choose_action(state, q_value, maze):
    if np.random.binomial(1, DynaQParams.epsilon) == 1:
        return np.random.choice(maze.ACTIONS)
    else:
        values = q_value[state[0], state[1], :]
        return np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])


# Dyna-Q의 계획 과정에서 사용하는 간단한 모델
class SimpleModel:
    def __init__(self):
        self.model = dict()

    # 경험 샘플 저장
    def store(self, state, action, reward, next_state):
        if state not in self.model:
            self.model[state] = dict()
        self.model[state][action] = [reward, next_state]

    # 저장해 둔 경험 샘플들에서 임으로 선택하여 반환
    def sample(self):
        state = random.choice(list(self.model.keys()))
        action = random.choice(list(self.model[state].keys()))
        reward, next_state = self.model[state][action]
        return state, action, next_state, reward


# Time-based model for planning in Dyna-Q+
class TimeModel:
    # @maze: the maze instance. Indeed it's not very reasonable to give access to maze to the model.
    # @timeWeight: also called kappa, the weight for elapsed time in sampling reward, it need to be small
    def __init__(self, maze, time_weight=1e-4):
        self.model = dict()

        # track the total time
        self.time = 0

        self.time_weight = time_weight
        self.maze = maze

    # feed the model with previous experience
    def store(self, state, action, reward, next_state):
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        self.time += 1
        if state not in self.model.keys():
            self.model[state] = dict()

            # Actions that had never been tried before from a state were allowed to be considered in the planning step
            for action_ in self.maze.ACTIONS:
                if action_ != action:
                    # Such actions would lead back to the same state with a reward of zero
                    # Notice that the minimum time stamp is 1 instead of 0
                    self.model[state][action_] = [list(state), 0, 1]

        self.model[state][action] = [list(next_state), reward, self.time]

    # randomly sample from previous experience
    def sample(self):
        state_index = np.random.choice(range(len(self.model.keys())))
        state = list(self.model)[state_index]
        action_index = np.random.choice(range(len(self.model[state].keys())))
        action = list(self.model[state])[action_index]
        next_state, reward, time = self.model[state][action]

        # adjust reward with elapsed time since last vist
        reward += self.time_weight * np.sqrt(self.time - time)

        state = deepcopy(state)
        next_state = deepcopy(next_state)

        return list(state), action, list(next_state), reward


# Model containing a priority queue for Prioritized Sweeping
class PriorityModel(SimpleModel):
    def __init__(self):
        SimpleModel.__init__(self)
        # maintain a priority queue
        self.priority_queue = PriorityQueue()
        # track predecessors for every state
        self.predecessors = dict()

    # add a @state-@action pair into the priority queue with priority @priority
    def insert(self, priority, state, action):
        # note the priority queue is a minimum heap, so we use -priority
        self.priority_queue.add_item((state, action), -priority)

    # @return: whether the priority queue is empty
    def is_empty(self):
        return self.priority_queue.is_empty()

    # get the first item in the priority queue
    def sample(self):
        (state, action), priority = self.priority_queue.pop_item()
        next_state, reward = self.model[state][action]
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        return -priority, list(state), action, list(next_state), reward

    # feed the model with previous experience
    def store(self, state, action, reward, next_state):
        state = deepcopy(state)
        next_state = deepcopy(next_state)
        SimpleModel.store(self, state, action, reward, next_state)
        if tuple(next_state) not in self.predecessors.keys():
            self.predecessors[tuple(next_state)] = set()
        self.predecessors[tuple(next_state)].add((state, action))

    # get all seen predecessors of a state @state
    def predecessor(self, state):
        if state not in self.predecessors.keys():
            return []
        predecessors = []
        for state_pre, action_pre in list(self.predecessors[state]):
            predecessors.append([list(state_pre), action_pre, self.model[state_pre][action_pre][1]])
        return predecessors


# play for an episode for Dyna-Q algorithm
# @q_value: state action pair values, will be updated
# @model: model instance for planning
# @maze: a maze instance containing all information about the environment
# @DynaQParams: several params for the algorithm
def dyna_q(q_value, model, maze):
    state = maze.START_STATE
    steps = 0
    while state not in maze.GOAL_STATES:
        # track the steps
        steps += 1

        # get action
        action = choose_action(state, q_value, maze)

        # take action
        next_state, reward = maze.step(state, action)

        # Q-Learning update
        target = reward + DynaQParams.gamma * np.max(q_value[next_state[0], next_state[1], :])
        q_value[state[0], state[1], action] += DynaQParams.alpha * (target - q_value[state[0], state[1], action])

        # store the model with experience
        model.store(state, action, reward, next_state)

        # sample experience from the model
        for t in range(0, DynaQParams.planning_steps):
            state_, action_, next_state_, reward_ = model.sample()
            target = reward_ + DynaQParams.gamma * np.max(q_value[next_state_[0], next_state_[1], :])
            q_value[state_[0], state_[1], action_] += DynaQParams.alpha * (target - q_value[state_[0], state_[1], action_])

        state = next_state

        # check whether it has exceeded the step limit
        if steps > maze.max_steps:
            break

    return steps


# play for an episode for prioritized sweeping algorithm
# @q_value: state action pair values, will be updated
# @model: model instance for planning
# @maze: a maze instance containing all information about the environment
# @DynaQParams: several params for the algorithm
# @return: # of backups during this episode
def prioritized_sweeping(q_value, model, maze):
    state = maze.START_STATE

    # track the steps in this episode
    steps = 0

    # track the backups in planning phase
    backups = 0

    while state not in maze.GOAL_STATES:
        steps += 1

        # get action
        action = choose_action(state, q_value, maze)

        # take action
        next_state, reward = maze.step(state, action)

        # feed the model with experience
        model.store(state, action, reward, next_state)

        # get the priority for current state action pair
        priority = np.abs(reward + DynaQParams.gamma * np.max(q_value[next_state[0], next_state[1], :]) -
                          q_value[state[0], state[1], action])

        if priority > DynaQParams.theta:
            model.insert(priority, state, action)

        # start planning
        planning_step = 0

        # planning for several steps,
        # although keep planning until the priority queue becomes empty will converge much faster
        while planning_step < DynaQParams.planning_steps and not model.empty():
            # get a sample with highest priority from the model
            priority, state_, action_, next_state_, reward_ = model.sample()

            # update the state action value for the sample
            delta = reward_ + DynaQParams.gamma * np.max(q_value[next_state_[0], next_state_[1], :]) - \
                    q_value[state_[0], state_[1], action_]
            q_value[state_[0], state_[1], action_] += DynaQParams.alpha * delta

            # deal with all the predecessors of the sample state
            for state_pre, action_pre, reward_pre in model.predecessor(state_):
                priority = np.abs(reward_pre + DynaQParams.gamma * np.max(q_value[state_[0], state_[1], :]) -
                                  q_value[state_pre[0], state_pre[1], action_pre])
                if priority > DynaQParams.theta:
                    model.insert(priority, state_pre, action_pre)
            planning_step += 1

        state = next_state

        # update the # of backups
        backups += planning_step + 1

    return backups


# 행동 가치함수를 표 형태로 그리는 함수
def draw_image(dyna_maze, q_value, run, planning_step, episode):
    # 축 표시 제거, 크기 조절 등 이미지 그리기 이전 설정 작업
    fig, axis = plt.subplots()
    axis.set_axis_off()
    table = Table(axis, bbox=[0, 0, 1, 1])

    num_rows, num_cols = dyna_maze.MAZE_HEIGHT, dyna_maze.MAZE_WIDTH
    width, height = 1.0 / num_cols, 1.0 / num_rows

    for i in range(dyna_maze.MAZE_HEIGHT):
        for j in range(dyna_maze.MAZE_WIDTH):
            if np.sum(q_value[i][j]) == 0.0:
                symbol = " "
            else:
                action_idx = np.argmax(q_value[i][j])
                symbol = dyna_maze.ACTION_SYMBOLS[action_idx]
            table.add_cell(i, j, width, height, text=symbol, loc='center', facecolor='white')

    # 행, 열 라벨 추가
    for i in range(dyna_maze.MAZE_HEIGHT):
        table.add_cell(i, -1, width, height, text=i, loc='right', edgecolor='none', facecolor='none')

    for j in range(dyna_maze.MAZE_WIDTH):
        table.add_cell(-1, j, width, height/2, text=j, loc='center', edgecolor='none', facecolor='none')

    for key, cell in table.get_celld().items():
         cell.get_text().set_fontsize(20)

    axis.add_table(table)
    plt.savefig('images/maze_action_values_{0}_{1}_{2}.png'.format(run, planning_step, episode))
    plt.close()


# DynaMaze, use 10 runs instead of 30 runs
def maze_dyna_q():
    # set up an instance for DynaMaze
    dyna_maze = Maze()

    episodes = 30
    planning_steps = [0, 3, 30]
    steps = np.zeros((len(planning_steps), episodes))

    for run in tqdm(range(DynaQParams.runs)):
        for i, planning_step in enumerate(planning_steps):
            DynaQParams.planning_steps = planning_step
            q_value = np.zeros(dyna_maze.q_size)

            # generate an instance of Dyna-Q model
            model = SimpleModel()
            for episode in range(episodes):
                #print('run:', run, 'planning step:', planning_step, 'episode:', episode)
                steps[i, episode] += dyna_q(q_value, model, dyna_maze)
                if run == 0 and planning_step in [0, 30] and episode in [0, 1]:
                    draw_image(dyna_maze, q_value, run, planning_step, episode)

    # 총 수행 횟수에 대한 평균 값 산출
    steps /= DynaQParams.runs

    linestyles = ['-', '--', ':']
    for i in range(len(planning_steps)):
        plt.plot(steps[i, :], linestyle=linestyles[i], label='계획시 수행 스텝: {0}'.format(planning_steps[i]))

    plt.xlabel('에피소드')
    plt.ylabel('에피소드당 진행 스텝수')
    plt.legend()

    plt.savefig('images/maze_dyna_q.png')
    plt.close()


# wrapper function for changing maze
# @maze: a maze instance
# @dynaParams: several parameters for dyna algorithms
def changing_maze(maze):

    # set up max steps
    max_steps = maze.max_steps

    # track the cumulative rewards
    rewards = np.zeros((DynaQParams.runs, 2, max_steps))

    for run in tqdm(range(DynaQParams.runs)):
        # set up models
        models = [SimpleModel(), TimeModel(maze, time_weight=DynaQParams.time_weight)]

        # initialize state action values
        q_values = [np.zeros(maze.q_size), np.zeros(maze.q_size)]

        for i in range(len(DynaQParams.methods)):
            # print('run:', run, DynaQParams.methods[i])

            # set old obstacles for the maze
            maze.obstacles = maze.old_obstacles

            steps = 0
            last_steps = steps
            while steps < max_steps:
                # play for an episode
                steps += dyna_q(q_values[i], models[i], maze)

                # update cumulative rewards
                rewards[run, i, last_steps: steps] = rewards[run, i, last_steps]
                rewards[run, i, min(steps, max_steps - 1)] = rewards[run, i, last_steps] + 1
                last_steps = steps

                if steps > maze.obstacle_switch_time:
                    # change the obstacles
                    maze.obstacles = maze.new_obstacles

    # averaging over runs
    rewards = rewards.mean(axis=0)

    return rewards

# Figure 8.4, BlockingMaze
def figure_8_4():
    # set up a blocking maze instance
    blocking_maze = Maze()
    blocking_maze.START_STATE = [5, 3]
    blocking_maze.GOAL_STATES = [[0, 8]]
    blocking_maze.old_obstacles = [[3, i] for i in range(0, 8)]

    # new obstalces will block the optimal path
    blocking_maze.new_obstacles = [[3, i] for i in range(1, 9)]

    # step limit
    blocking_maze.max_steps = 3000

    # obstacles will change after 1000 steps
    # the exact step for changing will be different
    # However given that 1000 steps is long enough for both algorithms to converge,
    # the difference is guaranteed to be very small
    blocking_maze.obstacle_switch_time = 1000

    # set up parameters
    DynaQParams.alpha = 1.0
    DynaQParams.planning_steps = 10
    DynaQParams.runs = 20

    # kappa must be small, as the reward for getting the goal is only 1
    DynaQParams.time_weight = 1e-4

    # play
    rewards = changing_maze(blocking_maze)

    for i in range(len(DynaQParams.methods)):
        plt.plot(rewards[i, :], label=DynaQParams.methods[i])
    plt.xlabel('time steps')
    plt.ylabel('cumulative reward')
    plt.legend()

    plt.savefig('images/figure_8_4.png')
    plt.close()

# Figure 8.5, ShortcutMaze
def figure_8_5():
    # set up a shortcut maze instance
    shortcut_maze = Maze()
    shortcut_maze.START_STATE = [5, 3]
    shortcut_maze.GOAL_STATES = [[0, 8]]
    shortcut_maze.old_obstacles = [[3, i] for i in range(1, 9)]

    # new obstacles will have a shorter path
    shortcut_maze.new_obstacles = [[3, i] for i in range(1, 8)]

    # step limit
    shortcut_maze.max_steps = 6000

    # obstacles will change after 3000 steps
    # the exact step for changing will be different
    # However given that 3000 steps is long enough for both algorithms to converge,
    # the difference is guaranteed to be very small
    shortcut_maze.obstacle_switch_time = 3000

    # 50-step planning
    DynaQParams.planning_steps = 50
    DynaQParams.runs = 5
    DynaQParams.time_weight = 1e-3
    DynaQParams.alpha = 1.0

    # play
    rewards = changing_maze(shortcut_maze)

    for i in range(len(DynaQParams.methods)):
        plt.plot( rewards[i, :], label=DynaQParams.methods[i])
    plt.xlabel('time steps')
    plt.ylabel('cumulative reward')
    plt.legend()

    plt.savefig('images/figure_8_5.png')
    plt.close()


# Check whether state-action values are already optimal
def check_path(q_values, maze):
    # get the length of optimal path
    # 14 is the length of optimal path of the original maze
    # 1.2 means it's a relaxed optifmal path
    max_steps = 14 * maze.resolution * 1.2
    state = maze.START_STATE
    steps = 0
    while state not in maze.GOAL_STATES:
        action = np.argmax(q_values[state[0], state[1], :])
        state, _ = maze.step(state, action)
        steps += 1
        if steps > max_steps:
            return False
    return True


# Example 8.4, mazes with different resolution
def example_8_4():
    # get the original 6 * 9 maze
    original_maze = Maze()

    # set up the parameters for each algorithm
    DynaQParams.planning_steps = 5
    DynaQParams.alpha = 0.5
    DynaQParams.gamma = 0.95

    params_prioritized = DynaQParams()
    params_prioritized.theta = 0.0001
    params_prioritized.planning_steps = 5
    params_prioritized.alpha = 0.5
    params_prioritized.gamma = 0.95

    params = [params_prioritized, DynaQParams]

    # set up models for planning
    models = [PriorityModel, SimpleModel]
    method_names = ['Prioritized Sweeping', 'Dyna-Q']

    # due to limitation of my machine, I can only perform experiments for 5 mazes
    # assuming the 1st maze has w * h states, then k-th maze has w * h * k * k states
    num_of_mazes = 5

    # build all the mazes
    mazes = [original_maze.extend_maze(i) for i in range(1, num_of_mazes + 1)]
    methods = [prioritized_sweeping, dyna_q]

    # My machine cannot afford too many runs...
    runs = 5

    # track the # of backups
    backups = np.zeros((runs, 2, num_of_mazes))

    for run in range(0, runs):
        for i in range(0, len(method_names)):
            for mazeIndex, maze in zip(range(0, len(mazes)), mazes):
                print('run %d, %s, maze size %d' % (run, method_names[i], maze.WORLD_HEIGHT * maze.WORLD_WIDTH))

                # initialize the state action values
                q_value = np.zeros(maze.q_size)

                # track steps / backups for each episode
                steps = []

                # generate the model
                model = models[i]()

                # play for an episode
                while True:
                    steps.append(methods[i](q_value, model, maze, params[i]))

                    # print best actions w.r.t. current state-action values
                    # printActions(currentStateActionValues, maze)

                    # check whether the (relaxed) optimal path is found
                    if check_path(q_value, maze):
                        break

                # update the total steps / backups for this maze
                backups[run, i, mazeIndex] = np.sum(steps)

    backups = backups.mean(axis=0)

    # Dyna-Q performs several backups per step
    backups[1, :] *= DynaQParams.planning_steps + 1

    for i in range(0, len(method_names)):
        plt.plot(np.arange(1, num_of_mazes + 1), backups[i, :], label=method_names[i])
    plt.xlabel('maze resolution factor')
    plt.ylabel('backups until optimal solution')
    plt.yscale('log')
    plt.legend()

    plt.savefig('images/example_8_4.png')
    plt.close()


if __name__ == '__main__':
    maze_dyna_q()
    # figure_8_4()
    # figure_8_5()
    # example_8_4()
