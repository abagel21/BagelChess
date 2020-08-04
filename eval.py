# WIP, unsure if will use convolutional neural network 
# or classical evaluation function for board state

# returns a value 1 to -1
# positive if favorable for engine
# negative if unfavorable
def eval(board, side):
    # checkmate
    if(board.is_checkmate()) :
        return -1 if board.turn else 1
    if(board.is_stalemate()):
        return 0
    # first check material advantage

    # then check positional advantage
    # using different positions for beginning and endgame
