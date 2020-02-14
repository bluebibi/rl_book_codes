import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.table import Table
from tqdm import tqdm
import heapq
from copy import deepcopy
import random

from chapter_08.maze import Maze, ChangingMaze

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
def choose_action(state, q_value, dyna_maze):
    if np.random.binomial(1, DynaQParams.epsilon) == 1:
        return np.random.choice(dyna_maze.ACTIONS)
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

    # 저장해 둔 경험 샘플들에서 임으로 선택하여 반환험
    def sample(self):
        state = random.choice(list(self.model.keys()))
        action = random.choice(list(self.model[state].keys()))
        reward, next_state = self.model[state][action]
        return state, action, reward, next_state


# Dyna-Q+를 위한 시간 기반 모델
class TimeModel:
    # @all_actions: 모든 행동 집합 [UP, DOWN, LEFT, RIGHT]
    # @time_weight: 샘플링된 보상에 대하여 경과 시간에 대한 가중치
    def __init__(self, all_actions, time_weight=1e-4):
        self.model = dict()

        # 전체 타임 스텝 진행 기록
        self.time = 0
        self.time_weight = time_weight
        self.all_actions = all_actions

    # 경험 샘플 저장
    def store(self, state, action, reward, next_state):
        self.time += 1

        if state not in self.model:
            self.model[state] = dict()

            # 임의의 상태에 대하여 선택되어지지 않은 행동들에 대해서도 계획 단계에 고려가 되도록 함
            for action_ in self.all_actions:
                if action_ != action:
                    # 그러한 행동들은 행동 수행 이후에 상태가 변동 없으며 보상은 0로 설정
                    # 타임 스텝의 최소값은 0이 아니라 1
                    self.model[state][action_] = [0, state, 1]

        self.model[state][action] = [reward, next_state, self.time]

    # 저장해 둔 경험 샘플들에서 임으로 선택하여 반환
    def sample(self):
        state = random.choice(list(self.model.keys()))
        action = random.choice(list(self.model[state].keys()))
        reward, next_state, time = self.model[state][action]

        # 모델에 기록된 방문 시각과 현재 시각 사이의 차가 클 수록,
        # 즉 오래전에 기록된 모델 정보일 수록 reward가 커지도록 조정함.
        # 따라서, 오래전에 경험해보고 최근에 경험해보지 못한 상태-행동 쌍이 좀 더 자주 발현되도록 유도함
        reward += self.time_weight * np.sqrt(self.time - time)

        return state, action, reward, next_state


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


# Dyna-Q 알고리즘의 각 에피소드 별 학습
# @q_value: 행동 가치 테이블, dyna_q 함수 수행 후 값이 갱신 됨
# @model: 계획시 사용할 모델
# @dyna_maze: 미로 환경
def dyna_q(q_value, model, dyna_maze):
    state = dyna_maze.START_STATE
    steps = 0
    cumulative_rewards = 0.0
    while state not in dyna_maze.GOAL_STATES:
        # 타임 스텝 기록
        steps += 1

        # 행동 얻어오기
        action = choose_action(state, q_value, dyna_maze)

        # 행동 수행후
        reward, next_state = dyna_maze.step(state, action)

        # Q-러닝 갱신
        target = reward + DynaQParams.gamma * np.max(q_value[next_state[0], next_state[1], :])
        q_value[state[0], state[1], action] += DynaQParams.alpha * (target - q_value[state[0], state[1], action])

        # 경험 샘플을 모델에 저장 (모델 구성)
        model.store(state, action, reward, next_state)

        # 모델로 부터 샘플 얻어오면서 Q-계획 반복 수행
        for t in range(0, DynaQParams.planning_steps):
            state_, action_, reward_, next_state_ = model.sample()
            target = reward_ + DynaQParams.gamma * np.max(q_value[next_state_[0], next_state_[1], :])
            q_value[state_[0], state_[1], action_] += DynaQParams.alpha * (target - q_value[state_[0], state_[1], action_])

        state = next_state
        cumulative_rewards += reward

        # 최대 스텝 체크
        if steps > dyna_maze.max_steps:
            break

    return steps, cumulative_rewards


# play for an episode for prioritized sweeping algorithm
# @q_value: state action pair values, will be updated
# @model: model instance for planning
# @dyna_maze: a dyna_maze instance containing all information about the environment
# @DynaQParams: several params for the algorithm
# @return: # of backups during this episode
def prioritized_sweeping(q_value, model, dyna_maze):
    state = dyna_maze.START_STATE

    # track the steps in this episode
    steps = 0

    # track the backups in planning phase
    backups = 0

    while state not in dyna_maze.GOAL_STATES:
        steps += 1

        # get action
        action = choose_action(state, q_value, dyna_maze)

        # take action
        reward, next_state = dyna_maze.step(state, action)

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

    # for i in range(dyna_maze.MAZE_HEIGHT):
    #     print("---------------------------------")
    #     out = '| '
    #     for j in range(dyna_maze.MAZE_WIDTH):
    #         t = ["{0:.1f}".format(x) for x in q_value[i][j]]
    #         out += str("{0}".format(t)) + ' | '
    #     print(out)
    # print("---------------------------------\n")


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
                steps_, _ = dyna_q(q_value, model, dyna_maze)
                steps[i, episode] += steps_
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


# @dyna_maze: a dyna_maze instance
def maze_dyna_q_2(dyna_maze):
    # 누적 보상을 기록하는 자료 구조
    cumulative_rewards = np.zeros((DynaQParams.runs, len(DynaQParams.methods), dyna_maze.max_steps))

    for run in tqdm(range(DynaQParams.runs)):
        # 두 개의 모델 설정
        models = [SimpleModel(), TimeModel(dyna_maze.ACTIONS, time_weight=DynaQParams.time_weight)]

        # 두 개의 모델별 행동 가치 초기화
        q_values = [np.zeros(dyna_maze.q_size), np.zeros(dyna_maze.q_size)]

        for method_idx in range(len(DynaQParams.methods)):
            # print('run:', run, DynaQParams.methods[method_idx])
            # set old obstacles for the maze
            dyna_maze.obstacles = dyna_maze.original_obstacles

            steps = 0
            last_steps = steps
            while steps < dyna_maze.max_steps:
                # play for an episode
                steps_, cumulative_rewards_ = dyna_q(q_values[method_idx], models[method_idx], dyna_maze)
                steps += steps_

                # update cumulative rewards
                cumulative_rewards[run, method_idx, last_steps: steps] = cumulative_rewards[run, method_idx, last_steps]
                cumulative_rewards[run, method_idx, min(steps, dyna_maze.max_steps - 1)] = cumulative_rewards[run, method_idx, last_steps] + cumulative_rewards_
                last_steps = steps

                if steps > dyna_maze.obstacle_switch_time:
                    # 장애물 구성 변경
                    dyna_maze.obstacles = dyna_maze.new_obstacles

                # print(DynaQParams.methods[method_idx], steps, dyna_maze.obstacles)

    # averaging over runs
    rewards = cumulative_rewards.mean(axis=0)

    return rewards


def changing_maze_dyna_q():
    # 장애물 구성 변경 미로 환경 구성
    changing_maze = ChangingMaze()
    changing_maze.original_obstacles = [(3, i) for i in range(0, 9)]

    # 최적 경로를 가로 막는 새로운 장애물 구성
    changing_maze.new_obstacles = [(3, i) for i in range(1, 10)]

    # 최대 스텝 설정
    changing_maze.max_steps = 3000

    # 장애물 구성이 변경되는 스텝 설정
    changing_maze.obstacle_switch_time = 1500

    # set up parameters
    DynaQParams.alpha = 1.0
    DynaQParams.planning_steps = 10
    DynaQParams.runs = 20

    # 시간 기반 모델에 사용할 파라미터
    DynaQParams.time_weight = 1e-4

    # 훈련 후 누적 보상 획득
    rewards = maze_dyna_q_2(changing_maze)

    for i in range(len(DynaQParams.methods)):
        plt.plot(rewards[i, :], label=DynaQParams.methods[i])

    plt.xlabel('타임 스텝')
    plt.ylabel('누적 보상')
    plt.legend()

    plt.savefig('images/changing_maze_dyna_q.png')
    plt.close()


# Figure 8.5, ShortcutMaze
def figure_8_5():
    # set up a shortcut dyna_maze instance
    shortcut_maze = ChangingMaze()
    shortcut_maze.START_STATE = [5, 3]
    shortcut_maze.GOAL_STATES = [[0, 8]]
    shortcut_maze.old_obstacles = [(3, i) for i in range(1, 9)]

    # new obstacles will have a shorter path
    shortcut_maze.new_obstacles = [(3, i) for i in range(1, 8)]

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
    rewards = maze_dyna_q_2(shortcut_maze)

    for i in range(len(DynaQParams.methods)):
        plt.plot( rewards[i, :], label=DynaQParams.methods[i])
    plt.xlabel('time steps')
    plt.ylabel('cumulative reward')
    plt.legend()

    plt.savefig('images/figure_8_5.png')
    plt.close()


# Check whether state-action values are already optimal
def check_path(q_values, dyna_maze):
    # get the length of optimal path
    # 14 is the length of optimal path of the original dyna_maze
    # 1.2 means it's a relaxed optifmal path
    max_steps = 14 * dyna_maze.resolution * 1.2
    state = dyna_maze.START_STATE
    steps = 0
    while state not in dyna_maze.GOAL_STATES:
        action = np.argmax(q_values[state[0], state[1], :])
        _, state = dyna_maze.step(state, action)
        steps += 1
        if steps > max_steps:
            return False
    return True


# Example 8.4, mazes with different resolution
def example_8_4():
    # get the original 6 * 9 dyna_maze
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
    # assuming the 1st dyna_maze has w * h states, then k-th dyna_maze has w * h * k * k states
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
            for mazeIndex, dyna_maze in zip(range(0, len(mazes)), mazes):
                print('run %d, %s, dyna_maze size %d' % (run, method_names[i], dyna_maze.WORLD_HEIGHT * dyna_maze.WORLD_WIDTH))

                # initialize the state action values
                q_value = np.zeros(dyna_maze.q_size)

                # track steps / backups for each episode
                steps = []

                # generate the model
                model = models[i]()

                # play for an episode
                while True:
                    steps.append(methods[i](q_value, model, dyna_maze, params[i]))

                    # print best actions w.r.t. current state-action values
                    # printActions(currentStateActionValues, dyna_maze)

                    # check whether the (relaxed) optimal path is found
                    if check_path(q_value, dyna_maze):
                        break

                # update the total steps / backups for this dyna_maze
                backups[run, i, mazeIndex] = np.sum(steps)

    backups = backups.mean(axis=0)

    # Dyna-Q performs several backups per step
    backups[1, :] *= DynaQParams.planning_steps + 1

    for i in range(0, len(method_names)):
        plt.plot(np.arange(1, num_of_mazes + 1), backups[i, :], label=method_names[i])
    plt.xlabel('dyna_maze resolution factor')
    plt.ylabel('backups until optimal solution')
    plt.yscale('log')
    plt.legend()

    plt.savefig('images/example_8_4.png')
    plt.close()


if __name__ == '__main__':
    #maze_dyna_q()
    changing_maze_dyna_q()
    # figure_8_5()
    # example_8_4()
