import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

plt.rcParams["font.family"] = 'NanumBarunGothic'
plt.rcParams["font.size"] = 32
mpl.rcParams['axes.unicode_minus'] = False

# 가능한 행동들: 히트, 스탠드
HIT = 0    # 새 카드 요청(히트).
STAND = 1  # 행동 종료(stand, 책에서는 stick).
ACTIONS = [HIT, STAND]

# 플레이어의 전략
# 현재 카드의 총합이 20, 21인 경우 STAND. 그 외엔 HIT
POLICY_OF_PLAYER = np.zeros(22, dtype=np.int)
for i in range(12, 20):
    POLICY_OF_PLAYER[i] = HIT
POLICY_OF_PLAYER[20] = POLICY_OF_PLAYER[21] = STAND


# 플레이어의 타깃 정책(target policy, off-policy에서 개선에 사용되는 정책)의 함수 형태
def target_policy_player(usable_ace_player, player_sum, dealer_card):
    return POLICY_OF_PLAYER[player_sum]


# 플레이어의 행위 정책(behavior policy, off-policy에서 행동 선택을 위한 정책)의 함수 형태
def behavior_policy_player(usable_ace_player, player_sum, dealer_card):
    if np.random.binomial(1, 0.5) == 1:
        return STAND
    return HIT


# 딜러의 전략
POLICY_OF_DEALER = np.zeros(22)
for i in range(12, 17):
    POLICY_OF_DEALER[i] = HIT
for i in range(17, 22):
    POLICY_OF_DEALER[i] = STAND


# 새로운 카드 획득
# 카드 수량은 무한하다고 가정
def get_card():
    card = np.random.randint(1, 14)
    card = min(card, 10)
    return card


# 카드의 숫자 반환(Ace는 11)
def card_value(card_id):
    return 11 if card_id == 1 else card_id


# 블랙 잭 게임 진행
# @policy_of_player: 플레이어를 위한 정책 지정
# @initial_state: 주어지는 초기 상태(플레이어 카드 내 사용가능 Ace 존재 여부, 플레이어 카드 총합, 딜러의 공개된 카드)
# @initial_action: 초기 행동
def play_black_jack(policy_of_player, initial_state=None, initial_action=None):
    # =============== 블랙 잭 게임 초기 설정 =============== #
    # 플레이어 소지 카드의 총합
    player_cards_sum = 0
    # 플레이어가 겪는 경험(상태, 행동 쌍)들을 저장할 변수
    player_experience_trajectory = []
    # 플레이어의 에이스 카드 11 사용 여부
    usable_ace_player = False

    # 딜러 관련 변수
    dealer_card1 = 0
    dealer_card2 = 0
    usable_ace_dealer = False

    if initial_state is None:
        # 임의의 초기 상태 생성
        while player_cards_sum < 12:
            # 플레이어 카드 총합이 12보다 낮다면 무조건 HIT 선택
            card = get_card()
            player_cards_sum += card_value(card)

            # 플레이어 카드 총합이 21을 넘긴 경우 11에서 1로 생각할 에이스 카드 존재 여부 확인
            if player_cards_sum > 21:
                assert player_cards_sum == 22
                player_cards_sum -= 10
            else:
                usable_ace_player = usable_ace_player | (1 == card)

        # 딜러에게 카드 전달. 첫 번째 카드를 공개한다고 가정
        dealer_card1 = get_card()
        dealer_card2 = get_card()

    else:
        # 지정된 초기 상태를 함수의 인자로 넘겨받은 경우
        usable_ace_player, player_cards_sum, dealer_card1 = initial_state
        dealer_card2 = get_card()

    # 게임의 상태
    state = [usable_ace_player, player_cards_sum, dealer_card1]

    # 딜러의 카드 총합 계산
    dealer_cards_sum = card_value(dealer_card1) + card_value(dealer_card2)
    usable_ace_dealer = 1 in (dealer_card1, dealer_card2)
    # 초기 상태에서 총합이 21을 넘기면 에이스가 2개인 것이므로 하나를 1로써 취급
    if dealer_cards_sum > 21:
        assert dealer_cards_sum == 22
        dealer_cards_sum -= 10
    assert dealer_cards_sum <= 21
    assert player_cards_sum <= 21

    # =============== 블랙 잭 게임 진행 =============== #
    # 플레이어의 차례
    while True:
        if initial_action is not None:
            action = initial_action
            initial_action = None
        else:
            # 현재 상태를 기반으로 행동 선택
            action = policy_of_player(usable_ace_player, player_cards_sum, dealer_card1)

        # 중요도 샘플링(Importance Sampling)을 위하여 플레이어의 경험 궤적을 추적
        player_experience_trajectory.append([(usable_ace_player, player_cards_sum, dealer_card1), action])

        if action == STAND:
            break
        elif action == HIT:
            new_card = get_card()

            # 플레이어가 가진 에이스 카드의 개수 추적
            player_ace_count = int(usable_ace_player)
            if new_card == 1:
                player_ace_count += 1
            player_cards_sum += card_value(new_card)

            # 버스트(bust)를 피하기 위해 에이스 카드가 있다면 1로써 취급
            while player_cards_sum > 21 and player_ace_count:
                player_cards_sum -= 10
                player_ace_count -= 1

            # 플레이어 버스트(bust)
            if player_cards_sum > 21:
                return state, -1, player_experience_trajectory

            assert player_cards_sum <= 21
            usable_ace_player = (player_ace_count == 1)

    # 딜러의 차례
    while True:
        action = POLICY_OF_DEALER[dealer_cards_sum]
        if action == STAND:
            break
        
        new_card = get_card()
        dealer_ace_count = int(usable_ace_dealer)
        if new_card == 1:
            dealer_ace_count += 1
        dealer_cards_sum += card_value(new_card)
        
        while dealer_cards_sum > 21 and dealer_ace_count:
            dealer_cards_sum -= 10
            dealer_ace_count -= 1
        
        if dealer_cards_sum > 21:
            return state, 1, player_experience_trajectory
        usable_ace_dealer = (dealer_ace_count == 1)

    # =============== 블랙 잭 게임 결과 판정 =============== #
    # 플레이어와 딜러 간의 카드 차이(게임 결과) 확인
    assert player_cards_sum <= 21 and dealer_cards_sum <= 21
    if player_cards_sum > dealer_cards_sum:
        return state, 1, player_experience_trajectory
    elif player_cards_sum == dealer_cards_sum:
        return state, 0, player_experience_trajectory
    else:
        return state, -1, player_experience_trajectory


def mc_on_policy():
    states_usable_ace_1, states_no_usable_ace_1 = monte_carlo_on_policy(10000)
    states_usable_ace_2, states_no_usable_ace_2 = monte_carlo_on_policy(500000)

    # 학습 결과를 이미지로 정리해서 저장하기 위한 코드
    states = [states_usable_ace_1,
              states_usable_ace_2,
              states_no_usable_ace_1,
              states_no_usable_ace_2]

    # 에이스 카드 사용 가능과 에피소드 횟수가 상이
    titles = ['Usable Ace, 10,000 에피소드',
              'Usable Ace, 500,000 에피소드',
              'No Usable Ace, 10,000 에피소드',
              'No Usable Ace, 500,000 에피소드']

    _, axes = plt.subplots(2, 2, figsize=(40, 30))
    plt.subplots_adjust(wspace=0.15, hspace=0.4)
    axes = axes.flatten()

    sns.set(font_scale=4)
    for state, title, axis in zip(states, titles, axes):
        fig = sns.heatmap(
            np.flipud(state),
            cmap="YlGnBu",
            ax=axis,
            xticklabels=range(1, 11),
            yticklabels=list(reversed(range(12, 22))),
            annot_kws={"size": 20},
            vmin=-1.0,
            vmax=1.0
        )
        fig.set_ylabel('플레이어 카드 합', fontsize=50)
        fig.set_xlabel('공개된 딜러 카드', fontsize=50)
        fig.set_title(title, fontsize=50, fontweight='bold')

    plt.savefig('images/monte_carlo_on_policy.png')
    plt.close()


# On-Policy로 작성된 몬테카를로 방법
def monte_carlo_on_policy(episodes):
    # 에이스 카드 사용 경우와 그렇지 않은 경우를 분리하여 생각
    states_usable_ace = np.zeros((10, 10))
    states_usable_ace_count = np.ones((10, 10))
    states_no_usable_ace = np.zeros((10, 10))
    states_no_usable_ace_count = np.ones((10, 10))

    for _ in tqdm(range(0, episodes)):
        # 플레이어 타깃 정책으로만 블랙 잭 게임을 진행하며 학습
        _, reward, player_experience_trajectory = play_black_jack(target_policy_player)
        for (usable_ace_player, player_cards_sum, dealer_card), _ in player_experience_trajectory:
            player_cards_sum -= 12
            dealer_card -= 1
            if usable_ace_player:
                states_usable_ace_count[player_cards_sum, dealer_card] += 1
                states_usable_ace[player_cards_sum, dealer_card] += reward
            else:
                states_no_usable_ace_count[player_cards_sum, dealer_card] += 1
                states_no_usable_ace[player_cards_sum, dealer_card] += reward
    return states_usable_ace / states_usable_ace_count, states_no_usable_ace / states_no_usable_ace_count


# 탐험적 시작 기법을 채용한 몬테카를로 방법
def monte_carlo_es(episodes):
    # 행동 가치 함수(플레이어 카드 총합, 공개된 딜러 카드, 에이스 카드 가용 여부, 행동)
    state_action_values = np.zeros((10, 10, 2, 2))
    # division by 0 에러를 막기 위해 1로 초기화
    state_action_pair_count = np.ones((10, 10, 2, 2))

    # 탐욕적 행위 정책
    def behavior_policy(usable_ace, player_cards_sum, dealer_card):
        usable_ace = int(usable_ace)
        player_cards_sum -= 12
        dealer_card -= 1
        # 주어진 상태 s 에서 가능한 행동 a 중 더 높은 가치를 갖는 행동을 계산하여 반환
        # 즉, argmax q(s, a) 를 게산하여 반환한다
        values_ = state_action_values[player_cards_sum, dealer_card, usable_ace, :] / state_action_pair_count[player_cards_sum, dealer_card, usable_ace, :]
        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

    # episodes 회의 에피소드 반복
    for episode in tqdm(range(episodes)):
        # 에피소드 마다 초기 설정은 무작위로 정한다
        initial_state = [bool(np.random.choice([0, 1])),
                       np.random.choice(range(12, 22)),
                       np.random.choice(range(1, 11))]
        initial_action = np.random.choice(ACTIONS)
        current_policy = behavior_policy if episode else target_policy_player   # episode = 0, 초기엔 타깃 정책 사용
        _, reward, trajectory = play_black_jack(current_policy, initial_state, initial_action)
        
        # 플레이어가 게임을 진행하며 방문한 경험으로 state_action_values와 state_action_pair_count 갱신
        for (usable_ace, player_cards_sum, dealer_card), action in trajectory:
            usable_ace = int(usable_ace)
            player_cards_sum -= 12
            dealer_card -= 1
            # 상태-행동 쌍에 대하여 값을 갱신
            state_action_values[player_cards_sum, dealer_card, usable_ace, action] += reward
            state_action_pair_count[player_cards_sum, dealer_card, usable_ace, action] += 1

    return state_action_values / state_action_pair_count


# Off-Policy로 작성된 몬테카를로 방법
def monte_carlo_off_policy(episodes):
    initial_state = [True, 13, 2]
    rhos = []
    returns = []

    for i in range(0, episodes):
        _, reward, player_experience_trajectory = play_black_jack(behavior_policy_player, initial_state=initial_state)

        # 중요도 비율을 얻는다
        numerator = 1.0     # 분자
        denominator = 1.0   # 분모
        for (usable_ace, player_cards_sum, dealer_card), action in player_experience_trajectory:
            if action == target_policy_player(usable_ace, player_cards_sum, dealer_card):
                denominator *= 0.5
            else:
                numerator = 0.0
                break
        rho = numerator / denominator
        rhos.append(rho)
        returns.append(reward)

    # 일반 중요도 샘플링(ordinary importance sampling) 과
    # 가중치 중요도 샘플링(weighted importance sampling) 각각을 계산하여 반환
    rhos = np.asarray(rhos)
    returns = np.asarray(returns)
    weighted_returns = rhos * returns

    weighted_returns = np.add.accumulate(weighted_returns)
    rhos = np.add.accumulate(rhos)

    # 일반 중요도 샘플링
    ordinary_importance_sampling = weighted_returns / np.arange(1, episodes + 1)

    # 가중치 중요도 샘플링
    with np.errstate(divide='ignore',invalid='ignore'):
        weighted_importance_sampling = np.where(rhos != 0, weighted_returns / rhos, 0)

    return ordinary_importance_sampling, weighted_importance_sampling


# 탐험적 시작 적용 몬테 카를로 방법
def mc_exploring_start():
    state_action_values = monte_carlo_es(500000)

    state_value_no_usable_ace = np.max(state_action_values[:, :, 0, :], axis=-1)
    state_value_usable_ace = np.max(state_action_values[:, :, 1, :], axis=-1)

    # argmax로 최적의 정책을 추출
    action_no_usable_ace = np.argmax(state_action_values[:, :, 0, :], axis=-1)
    action_usable_ace = np.argmax(state_action_values[:, :, 1, :], axis=-1)

    # 학습 결과를 이미지로 정리하여 저장하기 위한 코드
    images = [action_usable_ace,
              state_value_usable_ace,
              action_no_usable_ace,
              state_value_no_usable_ace]
    titles = ['Optimal policy with usable Ace',
              'Optimal value with usable Ace',
              'Optimal policy without usable Ace',
              'Optimal value without usable Ace']

    _, axes = plt.subplots(2, 2, figsize=(40, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()

    sns.set(font_scale=4)
    for image, title, axis in zip(images, titles, axes):
        fig = sns.heatmap(np.flipud(image), cmap="YlGnBu", ax=axis, xticklabels=range(1, 11),
                          yticklabels=list(reversed(range(12, 22))))
        fig.set_ylabel('player sum', fontsize=60)
        fig.set_xlabel('dealer showing', fontsize=50)
        fig.set_title(title, fontsize=50, fontweight='bold')

    plt.savefig('images/monte_carlo_exploring_start.png')
    plt.close()


# Off-Policy 몬테 카를로 방법
def mc_off_policy():
    # 변수들 초기화
    true_value = -0.27726
    episodes, runs = 10000, 100
    error_ordinary_sampling = np.zeros(episodes)
    error_weighted_sampling = np.zeros(episodes)
    for _ in tqdm(range(0, runs)):
        ordinary_importance_sampling_, weighted_importance_sampling_ = monte_carlo_off_policy(episodes)
        # 제곱 오차 계산 (일반 중요도 샘플링, 가중치 중요도 샘플링)
        error_ordinary_sampling += np.power(ordinary_importance_sampling_ - true_value, 2)
        error_weighted_sampling += np.power(weighted_importance_sampling_ - true_value, 2)
    error_ordinary_sampling /= runs
    error_weighted_sampling /= runs

    # 학습 결과를 이미지로 정리하여 저장하기 위한 코드
    sns.set(font_scale=1)
    plt.plot(error_ordinary_sampling, label='Ordinary Importance Sampling')
    plt.plot(error_weighted_sampling, label='Weighted Importance Sampling')
    plt.xlabel('Episodes (log scale)', fontsize=10)
    plt.ylabel('Mean square error', fontsize=10)
    plt.xscale('log')
    plt.legend()

    plt.savefig('images/monte_carlo_off_policy.png')
    plt.close()


if __name__ == '__main__':
    # Monte-Carlo On-policy, 탐험적 시작, Off-policy
    mc_on_policy()
    #mc_exploring_start()
    #mc_off_policy()
