# -*- coding:utf-8 -*-
"""
###############################################################################################
    10-Armed Testbed (Reinforcement Learning: An Introduction, Sutton, Barto, fig 2.2)
    Created by Youn-Hee Han 12/27/2019, last update 12/27/2019

    코드 참고 사이트:
    한글 폰트 설치 참고 사이트: https://programmers.co.kr/learn/courses/21/lessons/950
###############################################################################################
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib as mpl
from enum import Enum

plt.rcParams["font.family"] = 'NanumBarunGothic'
plt.rcParams["font.size"] = 12
mpl.rcParams['axes.unicode_minus'] = False


class ActionSelectionStrategy(Enum):
    greedy = 0
    epsilon_greedy = 1
    ucb = 2
    gradient = 3


class ValueUpdateStrategy(Enum):
    sample_average = 0
    incremental_sample_average = 1
    constant_step_size = 2
    gradient_with_baseline = 3
    gradient_without_baseline = 4


################################################################
# 상태와 행동 및 문제에 대한 설정 규칙을 포함하는 테스트베드 클래스
class Testbed(object):
    # Constructor
    def __init__(self, num_arms, mean, std, compensated_reward=0.0):
        # 암의 개수
        self.num_arms = num_arms

        # 참 가치와 보상 값 설정을 위한 정규 분포 파라미터
        self.mean = mean  # 평균
        self.std = std  # 표준 편차

        # 참 가치값 보정 값
        self.compensated_reward = compensated_reward

        self.q_true = None  # 행동에 대한 Value를 저장하는 배열
        self.best_action = 0  # 최적 Value를 저장하는 변수
        self.reset()

    # 매 시행시 맨 처음 불리우는 Reset 함수
    def reset(self):
        # 에이전트에게는 알려주지 않는 참 가치 값: 정규 분포(평균: 0, 표준 편차: 1)를 통해 샘플링
        self.q_true = np.random.randn(self.num_arms) + self.compensated_reward

        # 참 가치값이 가장 큰 행동 추출
        self.best_action = np.argmax(self.q_true)

    def __str__(self):
        msg = "참 가치: {0}, 최적 행동: {1}".format(
            self.q_true,
            self.best_action
        )
        return msg


################################################################
# 테스트베드 위에서 환경과 상호작용하는 에이전트
class Bandit(object):
    def __init__(self, num_arms,
                 action_selection_strategy, value_update_strategy,
                 q_estimation_initial=0.0, epsilon=0.0, ucb_param=0.0, constant_step_size=0.0):
        self.num_arms = num_arms    # 밴딧이 지닌 암의 개수
        self.action_selection_strategy = action_selection_strategy
        self.value_update_strategy = value_update_strategy

        self.epsilon = epsilon
        self.ucb_param = ucb_param
        self.constant_step_size = constant_step_size
        self.q_estimation_initial = q_estimation_initial

        self.time_step = None
        self.indices = None
        self.count_of_actions = None
        self.sum_of_reward = None
        self.action_prob = None
        self.average_reward = None
        self.q_estimation = None

        self.reset()

    def reset(self):
        self.time_step = 0
        self.indices = np.arange(self.num_arms)
        self.count_of_actions = np.zeros(self.num_arms)                     # 행동별 수행 횟수
        self.sum_of_reward = np.zeros(self.num_arms)
        self.action_prob = np.zeros(self.num_arms)
        self.average_reward = np.zeros(self.num_arms)
        self.q_estimation = np.zeros(self.num_arms) + self.q_estimation_initial  # 행동별 추정 가치

        ### ACTION SELECTION STRATEGY
        if self.action_selection_strategy == ActionSelectionStrategy.greedy:
            self.epsilon = 0.0
        elif self.action_selection_strategy == ActionSelectionStrategy.epsilon_greedy:
            assert self.epsilon > 0.0
        else:
            pass

        if self.action_selection_strategy == ActionSelectionStrategy.ucb:
            assert self.ucb_param > 0.0

        if self.action_selection_strategy == ActionSelectionStrategy.gradient:
            self.action_prob = np.zeros(self.num_arms)

        ### VALUE UPDATE STRATEGY
        if self.value_update_strategy == ValueUpdateStrategy.sample_average:
            self.sum_of_reward = np.zeros(self.num_arms)

        if self.value_update_strategy == ValueUpdateStrategy.constant_step_size:
            assert self.constant_step_size > 0.0

        if self.value_update_strategy == ValueUpdateStrategy.gradient_with_baseline:
            self.average_reward = np.zeros(self.num_arms)

    # Epsilon 탐욕적 선택 방법으로 행동 선택
    # 만약 Epsilon 값이 0이면 단순한 탐욕적 방법으로 행동이 선택됨
    def action(self):
        self.time_step += 1
        rand_prob = np.random.random()  # 0 ~ 1 사이의 임의의 값이 샘플링됨

        if self.action_selection_strategy in [ActionSelectionStrategy.greedy, ActionSelectionStrategy.epsilon_greedy]:
            if rand_prob < self.epsilon:
                selected_action = np.random.choice(self.indices)  # 임의의 행동 선택
            # 탐욕적 방법
            else:
                # 최고의 추정 가치에 해당하는 행동 인덱스를 가져옴
                # 추정하는 최대 가치에 해당하는 행동이 1개이면 그 행동을 선택
                # 추정하는 최대 가치에 해당하는 행동이 여러개이면 그러한 중 임의의 행동을 선택
                q_best = np.max(self.q_estimation)
                selected_action = np.random.choice(np.where(self.q_estimation == q_best)[0])
            return selected_action

        if self.action_selection_strategy == ActionSelectionStrategy.ucb:
            ucb_estimation = self.q_estimation + \
                             self.ucb_param * np.sqrt(np.log(self.time_step + 1) / (self.count_of_actions + 1e-5))
            q_best = np.max(ucb_estimation)
            return np.random.choice(np.where(ucb_estimation == q_best)[0])

        if self.action_selection_strategy == ActionSelectionStrategy.gradient:
            exp_est = np.exp(self.q_estimation)
            self.action_prob = exp_est / np.sum(exp_est)
            return np.random.choice(self.indices, p=self.action_prob)

    # 받아낸 보상 값을 통하여 추정 가치 갱신
    def step(self, selected_action, testbed):
        # 보상은 정규 분포(평균: q_*(at), 표준편차: 1)로 부터 샘플링함
        reward = np.random.normal(testbed.q_true[selected_action], scale=1)

        self.count_of_actions[selected_action] += 1     # 행동 수행 횟수 1 증가

        # 행동의 추정 가치 갱신
        if self.value_update_strategy == ValueUpdateStrategy.sample_average:
            self.sum_of_reward[selected_action] += reward  # 보상을 누적함
            self.q_estimation[selected_action] = self.sum_of_reward[selected_action] / self.count_of_actions[selected_action]

        if self.value_update_strategy == ValueUpdateStrategy.incremental_sample_average:
            self.q_estimation[selected_action] += (reward - self.q_estimation[selected_action]) / self.count_of_actions[selected_action]

        if self.value_update_strategy == ValueUpdateStrategy.constant_step_size:
            self.q_estimation[selected_action] += (reward - self.q_estimation[selected_action]) * self.constant_step_size

        if self.value_update_strategy in [ValueUpdateStrategy.gradient_with_baseline, ValueUpdateStrategy.gradient_without_baseline]:
            self.average_reward += (reward - self.average_reward) / self.time_step

            one_hot = np.zeros(self.num_arms)
            one_hot[selected_action] = 1

            if self.value_update_strategy == ValueUpdateStrategy.gradient_with_baseline:
                baseline = self.average_reward
            else:
                baseline = 0
            self.q_estimation += self.constant_step_size * (reward - baseline) * (one_hot - self.action_prob)

        return reward


################################################################
# 에이전트가 상호작용하는 환경 클래스
class Environment(object):
    def __init__(self, testbed, bandits, max_runs, max_time_steps):
        self.testbed = testbed
        self.bandits = bandits
        self.max_runs = max_runs
        self.max_time_steps = max_time_steps

    # 테스트 수행
    def play(self):
        # 각 에이전트 및 각 타입 스텝별로 누적 보상을 저장하는 배열
        cumulative_rewards = np.zeros((len(self.bandits), self.max_runs, self.max_time_steps))

        # 각 에이전트 및 각 타입 스텝별로 최적 행동 비율을 저장하는 배열
        optimal_actions = np.zeros((len(self.bandits), self.max_runs, self.max_time_steps))

        # 각 밴딧, 시행, 타임 스텝 별로 루프를 수행
        for bandit_idx, bandit in enumerate(self.bandits):
            for run in range(self.max_runs):
                # 매 100번의 시행마다 출력 시행 횟수 출력
                if (run % 100) == 0:
                    print("Bandit: {0} - Completed Runs: {1}".format(bandit_idx, run))

                # 테스트베드 초기화
                self.testbed.reset()
                bandit.reset()

                # 각 타입 스텝별로 루프를 수행
                for time_step in range(self.max_time_steps):
                    selected_action = bandit.action()

                    # 에이전트의 추정 가치 갱신 및 보상 얻어오기
                    reward = bandit.step(selected_action, self.testbed)

                    # 누적 보상 업데이트 (그래프 출력용)
                    cumulative_rewards[bandit_idx, run, time_step] = reward

                    # 테스트 베드에 설정된 에이전트에게 알려지지 않은 최적 행동과 동일한 행동을 수행하였는지 체크 (그래프 2에 표현됨)
                    if selected_action == self.testbed.best_action:
                        optimal_actions[bandit_idx, run, time_step] = 1

        mean_cumulative_rewards = cumulative_rewards.mean(axis=1)
        optimal_action_rates = optimal_actions.mean(axis=1)

        return mean_cumulative_rewards, optimal_action_rates


def draw_figure(bandits, labels, max_time_steps, target_values, xlabel, ylabel, filename):
    line_style = ['-', '--', '-.', ':']

    axis_x_point_step = int(max_time_steps / 200)

    for idx, _ in enumerate(bandits):
        plt.plot(
            [x + 1 for x in range(0, max_time_steps, axis_x_point_step)],
            target_values[idx][::axis_x_point_step],
            linestyle=line_style[idx],
            label=labels[idx]
        )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=4)
    plt.savefig(filename)
    plt.close()


def prob_distribution_of_rewards():
    plt.violinplot(dataset=np.random.randn(200, 10) + np.random.randn(10))
    plt.xlabel("밴딧의 행동")
    plt.ylabel("참 가치 값에 의한 보상의 확률분포")
    plt.savefig('images/prob_distribution_of_rewards.png')
    plt.close()


def greedy_and_epsilon_greedy(max_runs, max_time_steps):
    testbed = Testbed(num_arms=10, mean=0.0, std=1.0, compensated_reward=0.0)
    bandits = [
        Bandit(
            num_arms=10,
            action_selection_strategy=ActionSelectionStrategy.greedy,
            value_update_strategy=ValueUpdateStrategy.incremental_sample_average,
        ),
        Bandit(
            num_arms=10,
            action_selection_strategy=ActionSelectionStrategy.epsilon_greedy,
            value_update_strategy=ValueUpdateStrategy.incremental_sample_average,
            epsilon=0.2
        ),
        Bandit(
            num_arms=10,
            action_selection_strategy=ActionSelectionStrategy.epsilon_greedy,
            value_update_strategy=ValueUpdateStrategy.incremental_sample_average,
            epsilon=0.1
        ),
        Bandit(
            num_arms=10,
            action_selection_strategy=ActionSelectionStrategy.epsilon_greedy,
            value_update_strategy=ValueUpdateStrategy.incremental_sample_average,
            epsilon=0.01
        )
    ]

    env = Environment(testbed=testbed, bandits=bandits, max_runs=max_runs, max_time_steps=max_time_steps)
    mean_cumulative_rewards, optimal_action_rates = env.play()

    draw_figure(
        bandits=bandits,
        labels=[
            'epsilon = {0:4.2f}'.format(bandits[0].epsilon),
            'epsilon = {0:4.2f}'.format(bandits[1].epsilon),
            'epsilon = {0:4.2f}'.format(bandits[2].epsilon),
            'epsilon = {0:4.2f}'.format(bandits[3].epsilon)
        ],
        max_time_steps=max_time_steps,
        target_values=mean_cumulative_rewards,
        xlabel="타임 스텝",
        ylabel="평균 누적 보상",
        filename="images/greedy_and_epsilon_greedy_rewards-{0}.png".format(max_time_steps)
    )

    draw_figure(
        bandits=bandits,
        labels=[
            'epsilon = {0:4.2f}'.format(bandits[0].epsilon),
            'epsilon = {0:4.2f}'.format(bandits[1].epsilon),
            'epsilon = {0:4.2f}'.format(bandits[2].epsilon),
            'epsilon = {0:4.2f}'.format(bandits[3].epsilon)
        ],
        max_time_steps=max_time_steps,
        target_values=optimal_action_rates * 100,
        xlabel="타임 스텝",
        ylabel="최적 행동 비율(%)",
        filename="images/greedy_and_epsilon_greedy_optimal_actions-{0}.png".format(max_time_steps)
    )


def q_estimation_initial(max_runs, max_time_steps):
    testbed = Testbed(num_arms=10, mean=0.0, std=1.0, compensated_reward=0.0)
    bandits = [
        Bandit(
            num_arms=10,
            action_selection_strategy=ActionSelectionStrategy.greedy,
            value_update_strategy=ValueUpdateStrategy.incremental_sample_average,
            q_estimation_initial=5.0
        ),
        Bandit(
            num_arms=10,
            action_selection_strategy=ActionSelectionStrategy.epsilon_greedy,
            value_update_strategy=ValueUpdateStrategy.incremental_sample_average,
            epsilon=0.1,
            q_estimation_initial=0.0
        )
    ]

    env = Environment(testbed=testbed, bandits=bandits, max_runs=max_runs, max_time_steps=max_time_steps)
    mean_cumulative_rewards, optimal_action_rates = env.play()

    draw_figure(
        bandits=bandits,
        labels=[
            'epsilon = {0:4.2f}, initial_q_estimation = {1:3.1f}'.format(bandits[0].epsilon, 5.0),
            'epsilon = {0:4.2f}, initial_q_estimation = {1:3.1f}'.format(bandits[1].epsilon, 0.0),
        ],
        max_time_steps=max_time_steps,
        target_values=mean_cumulative_rewards,
        xlabel="타임 스텝",
        ylabel="평균 누적 보상",
        filename="images/q_estimation_initial_rewards.png"
    )

    draw_figure(
        bandits=bandits,
        labels=[
            'epsilon = {0:4.2f}, initial_q_estimation = {1:3.1f}'.format(bandits[0].epsilon, 5.0),
            'epsilon = {0:4.2f}, initial_q_estimation = {1:3.1f}'.format(bandits[1].epsilon, 0.0),
        ],
        max_time_steps=max_time_steps,
        target_values=optimal_action_rates * 100,
        xlabel="타임 스텝",
        ylabel="최적 행동 비율(%)",
        filename="images/q_estimation_initial_optimal_actions.png"
    )


def ucb_section(max_runs, max_time_steps):
    testbed = Testbed(num_arms=10, mean=0.0, std=1.0, compensated_reward=0.0)
    bandits = [
        Bandit(
            num_arms=10,
            action_selection_strategy=ActionSelectionStrategy.ucb,
            value_update_strategy=ValueUpdateStrategy.incremental_sample_average,
            ucb_param=2.0
        ),
        Bandit(
            num_arms=10,
            action_selection_strategy=ActionSelectionStrategy.epsilon_greedy,
            value_update_strategy=ValueUpdateStrategy.incremental_sample_average,
            epsilon=0.1
        )
    ]

    env = Environment(testbed=testbed, bandits=bandits, max_runs=max_runs, max_time_steps=max_time_steps)
    mean_cumulative_rewards, optimal_action_rates = env.play()

    draw_figure(
        bandits=bandits,
        labels=[
            'UCB selection (c = {0:3.1f})'.format(bandits[0].ucb_param),
            'epsilon greedy (epsilon = {0:4.2f})'.format(bandits[1].epsilon),
        ],
        max_time_steps=max_time_steps,
        target_values=mean_cumulative_rewards,
        xlabel="타임 스텝",
        ylabel="평균 누적 보상",
        filename="images/ucb_section_rewards.png"
    )

    draw_figure(
        bandits=bandits,
        labels=[
            'UCB selection (c = {0:3.1f})'.format(bandits[0].ucb_param),
            'epsilon greedy (epsilon = {0:4.2f})'.format(bandits[1].epsilon),
        ],
        max_time_steps=max_time_steps,
        target_values=optimal_action_rates * 100,
        xlabel="타임 스텝",
        ylabel="최적 행동 비율(%)",
        filename="images/ucb_section_optimal_actions.png"
    )


def gradients(max_runs, max_time_steps):
    testbed = Testbed(num_arms=10, mean=0.0, std=1.0, compensated_reward=4.0)
    bandits = [
        Bandit(
            num_arms=10,
            action_selection_strategy=ActionSelectionStrategy.gradient,
            value_update_strategy=ValueUpdateStrategy.gradient_with_baseline,
            constant_step_size=0.1
        ),
        Bandit(
            num_arms=10,
            action_selection_strategy=ActionSelectionStrategy.gradient,
            value_update_strategy=ValueUpdateStrategy.gradient_without_baseline,
            constant_step_size=0.1
        ),
        Bandit(
            num_arms=10,
            action_selection_strategy=ActionSelectionStrategy.gradient,
            value_update_strategy=ValueUpdateStrategy.gradient_with_baseline,
            constant_step_size=0.4
        ),
        Bandit(
            num_arms=10,
            action_selection_strategy=ActionSelectionStrategy.gradient,
            value_update_strategy=ValueUpdateStrategy.gradient_without_baseline,
            constant_step_size=0.4
        ),
    ]

    env = Environment(testbed=testbed, bandits=bandits, max_runs=max_runs, max_time_steps=max_time_steps)
    mean_cumulative_rewards, optimal_action_rates = env.play()

    draw_figure(
        bandits=bandits,
        labels=[
            'alpha = {0:3.1f}, with baseline'.format(bandits[0].constant_step_size),
            'alpha = {0:3.1f}, without baseline'.format(bandits[1].constant_step_size),
            'alpha = {0:3.1f}, with baseline'.format(bandits[2].constant_step_size),
            'alpha = {0:3.1f}, without baseline'.format(bandits[3].constant_step_size),
        ],
        max_time_steps=max_time_steps,
        target_values=mean_cumulative_rewards,
        xlabel="타임 스텝",
        ylabel="평균 누적 보상",
        filename="images/gradients_rewards.png"
    )

    draw_figure(
        bandits=bandits,
        labels=[
            'alpha = {0:3.1f}, with baseline'.format(bandits[0].constant_step_size),
            'alpha = {0:3.1f}, without baseline'.format(bandits[1].constant_step_size),
            'alpha = {0:3.1f}, with baseline'.format(bandits[2].constant_step_size),
            'alpha = {0:3.1f}, without baseline'.format(bandits[3].constant_step_size),
        ],
        max_time_steps=max_time_steps,
        target_values=optimal_action_rates * 100,
        xlabel="타임 스텝",
        ylabel="최적 행동 비율(%)",
        filename="images/gradients_optimal_actions.png"
    )


def comparison_of_all_methods(max_runs, max_time_steps):
    labels = ['epsilon-greedy', 'gradient bandit', 'UCB selection', 'optimistic initialization']
    testbed = Testbed(num_arms=10, mean=0.0, std=1.0, compensated_reward=4.0)
    generators = [
        lambda epsilon: Bandit(
            num_arms=10,
            action_selection_strategy=ActionSelectionStrategy.epsilon_greedy,
            value_update_strategy=ValueUpdateStrategy.incremental_sample_average,
            epsilon=epsilon
        ),
        lambda alpha: Bandit(
            num_arms=10,
            action_selection_strategy=ActionSelectionStrategy.gradient,
            value_update_strategy=ValueUpdateStrategy.gradient_with_baseline,
            constant_step_size=alpha
        ),
        lambda coef: Bandit(
            num_arms=10,
            action_selection_strategy=ActionSelectionStrategy.ucb,
            value_update_strategy=ValueUpdateStrategy.incremental_sample_average,
            ucb_param=coef
        ),
        lambda initial: Bandit(
            num_arms=10,
            action_selection_strategy=ActionSelectionStrategy.greedy,
            value_update_strategy=ValueUpdateStrategy.incremental_sample_average,
            q_estimation_initial=initial
        ),
    ]
    parameters = [
        np.arange(-7, -1, dtype=np.float),
        np.arange(-5, 2, dtype=np.float),
        np.arange(-4, 3, dtype=np.float),
        np.arange(-2, 3, dtype=np.float)
    ]

    bandits = []
    for generator, parameter in zip(generators, parameters):
        for param in parameter:
            bandits.append(generator(pow(2, param)))

    env = Environment(testbed=testbed, bandits=bandits, max_runs=max_runs, max_time_steps=max_time_steps)
    mean_cumulative_rewards, optimal_action_rates = env.play()

    i = 0
    for label, parameter in zip(labels, parameters):
        l = len(parameter)
        plt.plot(parameter, mean_cumulative_rewards[i:i+l], label=label)
        i += l
    plt.xlabel('Parameter(2^x)')
    plt.ylabel('Average reward')
    plt.legend()

    plt.savefig('images/comparison_of_all_methods.png')
    plt.close()


if __name__ == '__main__':
    # prob_distribution_of_rewards()
    # greedy_and_epsilon_greedy(max_runs=2000, max_time_steps=1000)
    # greedy_and_epsilon_greedy(max_runs=2000, max_time_steps=4000)
    #q_estimation_initial(max_runs=2000, max_time_steps=1000)
    ucb_section(max_runs=2000, max_time_steps=1000)
    #gradients(max_runs=2000, max_time_steps=1000)
    #comparison_of_all_methods(max_runs=2000, max_time_steps=1000)