import copy
import random
from exceptions import AgentException


def basic_static_eval(connect4, player):
    result = 0
    opponent = 'o' if player == 'x' else 'x'

    if connect4.game_over:
        if connect4.wins == player:
            return float('inf')
        elif connect4.wins == opponent:
            return -float('inf')
        else:
            return 0

    for four in connect4.iter_fours():
        player_count = four.count(player)
        opponent_count = four.count('o' if player == 'x' else 'x')

        if player_count == 3:
            result += 1
        elif opponent_count == 3:
            result -= 1
    return result

class MinMaxAgent:
    def __init__(self, my_token="o", heuristic_func=basic_static_eval):
        self.my_token = my_token
        self.heuristic_func = heuristic_func

    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException("not my round")

        best_move, best_score = self.minmax(connect4)
        return best_move

    def minmax(self, connect4, depth=4, maximizing=True) -> tuple[float, int]:
        if depth == 0 or connect4.game_over:
            return None, self.heuristic_func(connect4, self.my_token)

        if maximizing:
            best_score = -100000
            best_move = None

            possible_drops = connect4.possible_drops()  # Getting all possible moves
            random.shuffle(possible_drops)

            for move in possible_drops:
                next_connect4 = copy.deepcopy(connect4)  # Copying game state and -> to next move
                next_connect4.drop_token(move)
                _, score = self.minmax(next_connect4, depth-1, False)
                if score > best_score:
                    best_score = score
                    best_move = move
            return best_move, best_score
        else:
            best_score = 100000
            best_move = None

            possible_drops = connect4.possible_drops()
            random.shuffle(possible_drops)

            for move in possible_drops:
                next_connect4 = copy.deepcopy(connect4)
                next_connect4.drop_token(move)
                _, score = self.minmax(next_connect4, depth-1, True)
                if score < best_score:
                    best_score = score
                    best_move = move
            return best_move, best_score
