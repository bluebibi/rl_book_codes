from termcolor import colored

PLAYER_1 = 1  # Human
PLAYER_2 = 2  # Computer Agent


class SimpleTicTacToe:
    def __init__(self, player_just_moved):
        self.player_just_moved = player_just_moved
        self.current_state = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    def do_move(self, position):
        assert 0 <= position <= 8 and self.current_state[position] == 0
        self.player_just_moved = 3 - self.player_just_moved
        self.current_state[position] = self.player_just_moved

    def get_available_positions(self):
        if self.check_state() != 0:  # Winner가 결정되었거나 무승부이면
            return []
        else:
            available_positions = []
            for i in range(9):
                if self.current_state[i] == 0:
                    available_positions.append(i)

            return available_positions

    def get_result(self, player_just_moved):
        result = self.check_state()
        assert result != 0

        if result == -1:
            return 0.5

        elif result == player_just_moved:
            return 1.0
        else:
            return 0.0

    def check_state(self):
        for (x, y, z) in [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]:
            if self.current_state[x] == self.current_state[y] == self.current_state[z]:
                if self.current_state[x] == PLAYER_1:
                    return PLAYER_1
                elif self.current_state[x] == PLAYER_2:
                    return PLAYER_2

        if not [i for i in range(9) if self.current_state[i] == 0]:
            return -1
        return 0

    def __repr__(self):
        s = "-------------------\n"
        for i in range(3):
            for j in range(3):
                s += "|{0}    ".format(i * 3 + j)
            s += "|\n"

            for j in range(0, 3):
                s += "|  {0}  ".format(colored(" OX"[self.current_state[i * 3 + j]], "red"))
            s += "|\n"

            for j in range(3):
                s += "|     ".format(i * 3 + j)
            s += "|\n"

            s += "-------------------\n"
        return s