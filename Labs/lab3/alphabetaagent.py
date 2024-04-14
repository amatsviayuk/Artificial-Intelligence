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


def advanced_static_eval(connect4, player):
    player_score = 0
    opponent_score = 0
    for four in connect4.iter_fours():
        player_count = four.count(player)  # Pawns numbers
        opponent_count = four.count('o' if player == 'x' else 'x')

        if player_count == 4:  # Game over
            return 1
        elif opponent_count == 4:
            return -1

        if player_count == 3:
            player_score += 100
        elif player_count == 2:
            player_score += 10
        elif player_count == 1:
            player_score += 1

        elif opponent_count == 3:
            opponent_score += 100
        elif opponent_count == 2:
            opponent_score += 10
        elif opponent_count == 1:
            opponent_score += 1

    total_score = (player_score - opponent_score) / 100
    final_score = min(max(total_score, -1), 1)  # [-1,1]
    return final_score


class AlphaBetaAgent:
    def __init__(self, my_token="o", heuristic_func=advanced_static_eval):
        self.my_token = my_token
        self.heuristic_func = heuristic_func

    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException("not my round")

        best_move, best_score = self.alphabeta(connect4, 4, True, -100000, 10000)
        return best_move

    def alphabeta(self, connect4, depth=4, maximizing=True, alpha=0, beta=0) -> tuple[float, int]:
        if depth == 0 or connect4.game_over:
            return 0, self.heuristic_func(connect4, self.my_token)

        if maximizing:
            best_score = -float('inf')
            best_move = None

            possible_drops = connect4.possible_drops()  # Getting all possible moves
            random.shuffle(possible_drops)

            for move in possible_drops:
                next_connect4 = copy.deepcopy(connect4)  # Copying game state and -> to next move
                next_connect4.drop_token(move)
                _, score = self.alphabeta(next_connect4, depth-1, False, alpha, beta)
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, score)
                if alpha >= beta:
                    break
            return best_move, best_score
        else:
            best_score = float('inf')
            best_move = None

            possible_drops = connect4.possible_drops()
            random.shuffle(possible_drops)

            for move in possible_drops:
                next_connect4 = copy.deepcopy(connect4)
                next_connect4.drop_token(move)
                _, score = self.alphabeta(next_connect4, depth-1, True, alpha, beta)
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, score)
                if alpha >= beta:
                    break
            return best_move, best_score
