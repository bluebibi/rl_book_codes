from math import *
import random
import copy

from environments.tic_tac_toe_simple import SimpleTicTacToe
from utils.logger import get_logger

PLAYER_1 = 1  # Human
PLAYER_2 = 2  # Computer Agent

logger = get_logger("mctc")


class Node:
    def __init__(self, position=-1, parent=None, env=None):
        self.position = position
        self.parent_node = parent
        self.child_nodes = []
        self.wins = 0
        self.visits = 0
        self.untried_positions = env.get_available_positions()
        self.player_just_moved = env.player_just_moved

    def select_child_uct(self):
        s = sorted(self.child_nodes, key=lambda c: c.wins / c.visits + sqrt(2 * log(self.visits) / c.visits))
        return s[-1]

    def append_child(self, position, env):
        child_node = Node(position=position, parent=self, env=env)
        self.untried_positions.remove(position)
        self.child_nodes.append(child_node)
        return child_node

    def update(self, result):
        self.visits += 1
        self.wins += result

    def __repr__(self):
        return "[Position: {0:4}, Wins/Visits: {1:3.2f}/{2:3.2f}, Untried Positions: {3}]".format(
            self.position, self.wins, self.visits, self.untried_positions
        )

    def children_to_str(self):
        self.child_nodes.sort(key=lambda c: c.wins / c.visits)
        self.child_nodes.sort(key=lambda c: c.visits)
        str_ = ""
        for node in self.child_nodes:
            str_ += str(node) + "\n"
        return str_


def UCT(env, itermax):
    root_node = Node(env=env)

    for i in range(itermax):
        node_last_visited = root_node
        env_copied = copy.deepcopy(env)

        # Selection
        while not node_last_visited.untried_positions and node_last_visited.child_nodes:
            env_current_state = env_copied.current_state.copy()
            node_last_visited = node_last_visited.select_child_uct()
            env_copied.do_move(node_last_visited.position)
            logger.info("Iter {0:4}: Copied Environment State: {1} - {2:>15} - Node Last Visited: {3}".format(
                i, env_current_state, "Selection", node_last_visited
            ))

        # Expansion
        if node_last_visited.untried_positions:
            env_current_state = env_copied.current_state.copy()
            new_position = random.choice(node_last_visited.untried_positions)
            env_copied.do_move(new_position)
            node_last_visited = node_last_visited.append_child(new_position, env_copied)
            logger.info("Iter {0:4}: Copied Environment State: {1} - {2:>15} - Node Last Visited: {3}".format(
                i, env_current_state, "Expansion", node_last_visited
            ))

        # Simulation
        # [주의] Node는 새롭게 생성하지 않고 Copied Environment에서만 마지막 말단 노드까지 이동
        while env_copied.get_available_positions():
            env_current_state = env_copied.current_state.copy()
            env_copied.do_move(random.choice(env_copied.get_available_positions()))
            logger.info("Iter {0:4}: Copied Environment State: {1} - {2:>15} - Node Last Visited: {3}".format(
                i, env_current_state, "Simulation", node_last_visited
            ))

        # BackPropagation
        while node_last_visited is not None:
            env_current_state = env_copied.current_state.copy()
            node_last_visited.update(env_copied.get_result(node_last_visited.player_just_moved))
            logger.info("Iter {0:4}: Copied Environment State: {1} - {2:>15} - Node Last Visited: {3}".format(
                i, env_current_state, "Backpropagation", node_last_visited
            ))
            node_last_visited = node_last_visited.parent_node

        logger.info("")

    print(root_node.children_to_str())
    logger.info(root_node.children_to_str())

    s = sorted(root_node.child_nodes, key=lambda c: c.wins / c.visits)
    return sorted(s, key=lambda c: c.visits)[-1].position


def play_game_utc(human_move_first=True):
    player_just_moved = PLAYER_2 if human_move_first else PLAYER_1
    env = SimpleTicTacToe(player_just_moved=player_just_moved)

    while env.get_available_positions():
        print(env)

        if env.player_just_moved == PLAYER_1:
            new_position = UCT(env, itermax=3000)
        else:
            new_position = int(input("which position do you want to move? : "))
        print("Best Move : {0}\n".format(new_position))
        env.do_move(new_position)

    if env.get_result(env.player_just_moved) == 1.0:
        print(env)
        print("Player {0} Wins!".format(env.player_just_moved))

    elif env.get_result(env.player_just_moved) == 0.0:
        print(env)
        print("Payer {0} Wins!".format(3 - env.player_just_moved))

    else:
        print(env)
        print("Draw!!")


if __name__ == "__main__":
    play_game_utc(human_move_first=False)

