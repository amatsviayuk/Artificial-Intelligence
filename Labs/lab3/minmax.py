import random
from copy import deepcopy

from connect4 import Connect4


class MinMax:

    def __init__(self, my_token, max_depth=4):
        self.my_token = my_token
        self.max_depth = max_depth

    def moveGrade(self, connect4, player):
        player_score = 0
        opponent_score = 0
        for four in connect4.iter_fours():
            player_count = four.count(player)  # Pawns numbers
            opponent_count = four.count('o' if player == 'x' else 'x')

            if player_count == 4:  # Game over
                return 1
            elif opponent_count == 4:
                return -1

            if player_count == 3 and opponent_count == 0:
                player_score += 100
            elif player_count == 2 and opponent_count == 0:
                player_score += 10
            elif player_count == 1 and opponent_count == 0:
                player_score += 1
            elif opponent_count == 3 and player_count == 0:
                opponent_score += 100
            elif opponent_count == 2 and player_count == 0:
                opponent_score += 10
            elif opponent_count == 1 and player_count == 0:
                opponent_score += 1

        total_score = (player_score - opponent_score) / 100
        final_score = min(max(total_score, -1), 1)  # [-1,1]
        return final_score

    def value(self, connect4, player):  # Analysing the value
        grade = self.moveGrade(connect4, player)
        if grade > 0:
            return 1
        elif grade == 0:
            return 0
        else:
            return -1

    def result(self, connect4, player):  # Game result
        if player == 'x':  # Defining players
            opponent = 'o'
        else:
            opponent = 'x'

        if connect4.wins == player:  # Game state
            return 1
        elif connect4.wins == opponent:
            return -1
        else:
            return 0

    # Searching for the best move with MinMax
    def decide(self, connect4: Connect4) -> int:
        if self.max_depth == 0 or connect4.game_over:
            return
        return self.max_value(connect4, self.max_depth, self.my_token)[1]

    # Searching for player best move in current game state
    def max_value(self, connect4: Connect4, depth, player) -> tuple[float, int]:
        if depth == 0:  # No heuristic
            return 0, None
        if connect4.game_over:
            return self.result(connect4, player), None

        best_value = -100000
        best_action = None

        possible_drops = connect4.possible_drops()  # Getting possible moves
        random.shuffle(possible_drops)

        for action in possible_drops:
            next_state = deepcopy(connect4)  # Copying game state and -> to next move
            next_state.drop_token(action)
            value = self.min_value(next_state, depth-1, player)[0]  # Analysing opponent's move
            if value > best_value:
                best_value = value
                best_action = action

        return best_value, best_action

    # Searching for player best move in current game state
    def min_value(self, connect4, depth, player) -> tuple[float, int]:
        if depth == 0:
            return 0, None

        if connect4.game_over:
            return self.result(connect4, player), None

        best_value = 100000
        best_action = None

        possible_drops = connect4.possible_drops()
        random.shuffle(possible_drops)

        for action in possible_drops:
            next_state = deepcopy(connect4)
            next_state.drop_token(action)
            value = self.max_value(next_state, depth-1, player)[0]
            if value < best_value:
                best_value = value
                best_action = action

        return best_value, best_action


