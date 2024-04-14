from exceptions import GameplayException
from connect4 import Connect4
from minmax import MinMax

connect4 = Connect4(width=6, height=6)
bot = MinMax('o')

while not connect4.game_over:
    connect4.draw()
    try:
        if connect4.who_moves == bot.my_token:
            n_column = bot.decide(connect4)
            pass
        else:
            n_column = int(input(':'))
        connect4.drop_token(n_column)
    except (ValueError, GameplayException):
        print('invalid move')

connect4.draw()
