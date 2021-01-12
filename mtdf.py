import math
import numpy as np
import chess
import time
import functools
from zobristHashing import *

# hash dict for converting python-chess board to numpy array to feed to neural network
bithash = {
    "." : np.array([0,0,0,0,0,0]),
    "r" : np.array([0,0,0,1,0,0]),
    "b" : np.array([0,0,1,0,0,0]),
    "q" : np.array([0,0,0,0,1,0]),
    "k" : np.array([0,0,0,0,0,1]),
    "n" : np.array([0,1,0,0,0,0]),
    "p" : np.array([1,0,0,0,0,0]),
    "R" : np.array([0,0,0,-1,0,0]),
    "B" : np.array([0,0,-1,0,0,0]),
    "Q" : np.array([0,0,0,0,-1,0]),
    "K" : np.array([0,0,0,0,0,-1]),
    "N" : np.array([0,-1,0,0,0,0]),
    "P" : np.array([-1,0,0,0,0,0]),
}
engineModel = {}
input_details = {}
output_details = {}
globalBoard = {}

# dict for repeat game states
state_dict = dict()

# Piece-Square boards from sunfish algorithm (https://github.com/thomasahle/sunfish/blob/master/sunfish.py)
piece = { 1: 100, 2: 280, 3: 320, 4: 479, 5: 929, 6:60000 }
pst = {
    1: (   0,   0,   0,   0,   0,   0,   0,   0,
            78,  83,  86,  73, 102,  82,  85,  90,
             7,  29,  21,  44,  40,  31,  44,   7,
           -17,  16,  -2,  15,  14,   0,  15, -13,
           -26,   3,  10,   9,   6,   1,   0, -23,
           -22,   9,   5, -11, -10,  -2,   3, -19,
           -31,   8,  -7, -37, -36, -14,   3, -31,
             0,   0,   0,   0,   0,   0,   0,   0),
    2: ( -66, -53, -75, -75, -10, -55, -58, -70,
            -3,  -6, 100, -36,   4,  62,  -4, -14,
            10,  67,   1,  74,  73,  27,  62,  -2,
            24,  24,  45,  37,  33,  41,  25,  17,
            -1,   5,  31,  21,  22,  35,   2,   0,
           -18,  10,  13,  22,  18,  15,  11, -14,
           -23, -15,   2,   0,   2,   0, -23, -20,
           -74, -23, -26, -24, -19, -35, -22, -69),
    3: ( -59, -78, -82, -76, -23,-107, -37, -50,
           -11,  20,  35, -42, -39,  31,   2, -22,
            -9,  39, -32,  41,  52, -10,  28, -14,
            25,  17,  20,  34,  26,  25,  15,  10,
            13,  10,  17,  23,  17,  16,   0,   7,
            14,  25,  24,  15,   8,  25,  20,  15,
            19,  20,  11,   6,   7,   6,  20,  16,
            -7,   2, -15, -12, -14, -15, -10, -10),
    4: (  35,  29,  33,   4,  37,  33,  56,  50,
            55,  29,  56,  67,  55,  62,  34,  60,
            19,  35,  28,  33,  45,  27,  25,  15,
             0,   5,  16,  13,  18,  -4,  -9,  -6,
           -28, -35, -16, -21, -13, -29, -46, -30,
           -42, -28, -42, -25, -25, -35, -26, -46,
           -53, -38, -31, -26, -29, -43, -44, -53,
           -30, -24, -18,   5,  -2, -18, -31, -32),
    5: (   6,   1,  -8,-104,  69,  24,  88,  26,
            14,  32,  60, -10,  20,  76,  57,  24,
            -2,  43,  32,  60,  72,  63,  43,   2,
             1, -16,  22,  17,  25,  20, -13,  -6,
           -14, -15,  -2,  -5,  -1, -10, -20, -22,
           -30,  -6, -13, -11, -16, -11, -16, -27,
           -36, -18,   0, -19, -15, -15, -21, -38,
           -39, -30, -31, -13, -31, -36, -34, -42),
    6: (   4,  54,  47, -99, -99,  60,  83, -62,
           -32,  10,  55,  56,  56,  55,  10,   3,
           -62,  12, -57,  44, -67,  28,  37, -31,
           -55,  50,  11,  -4, -19,  13,   0, -49,
           -55, -43, -52, -28, -51, -47,  -8, -50,
           -47, -42, -43, -79, -64, -32, -29, -32,
            -4,   3, -14, -50, -57, -18,  13,   4,
            17,  30,  -3, -14,   6,  -1,  40,  18),
}
# Pad tables and join piece and pst dictionaries
for k, table in pst.items():
    padrow = lambda row: (0,) + tuple(x+piece[k] for x in row) + (0,)
    pst[k] = sum((padrow(table[i*8:i*8+8]) for i in range(8)), ())
    pst[k] = (0,)*20 + pst[k] + (0,)*20

def compareBoards(item1, item2) :
    state1 = predictFromMove(item1, globalBoard)
    state2 = predictFromMove(item2, globalBoard)
    return state1 - state2

def resetMem():
    global state_dict
    state_dict.clear()


A1, H1, A8, H8 = 56, 63, 0, 7

def predictFromMove(move, board):
    # TODO continue refining evaluation function based on move
    baseboard = board.board.copy()
    i = move.from_square
    j = move.to_square
    c = board.board.turn
    # if the move was made by black, mirror and flip the board, then adjust the move for the new transformed board
    if c:
        # print("is c")
        baseboard = baseboard.transform(chess.flip_vertical).mirror()
        # i = 64 - i
        # j = 64 - j
    res = 0
    p = baseboard.piece_type_at(i)
    # add the value of the new position of the piece - the value of the old position of the piece
    res += pst[p][j] - pst[p][i]
    # add the value of the capture if there was a capture
    if(baseboard.piece_type_at(j) is not None and not baseboard.color_at(j)):
        res += pst[baseboard.piece_type_at(j)][63 - j]
    # pawn promotion
    if p == 1:
            if A8 <= j <= H8:
                res += pst[5][j] - pst[0][j]
    return res

def predictFromBoard(node) :
        global leafCount
        global input_details
        global output_details
        global engineModel
        hash = node.zobrist
        if hash in state_dict: 
            leafCount = leafCount - 1
            return state_dict[hash]
        x = np.stack(np.array(boardToNPArray(node.board)))
        x = x.reshape(1,8,8,6) 
        # end = time.time()
        # print("Reshaping took " + str(end - start))
        # predict the value of the board state with the CNN
        # start = time.time()
        engineModel.set_tensor(input_details[0]['index'], x.astype(np.float32))
        engineModel.invoke()
        output_data = engineModel.get_tensor(output_details[0]['index'])
        results = np.squeeze(output_data)
        # end = time.time()
        # print("Predicting took " + str(end - start))
        state_dict[hash] = results
        return results

def boardToNPArray(x):
    x=x.__str__()
    x = x.split("\n")
    for n in range(len(x)):
        x[n] = np.array(x[n].split()).reshape(8,1)
        newTemp = []
        for i in range(len(x[n])) :
#             print(x[n][i][0])
             newTemp.append(bithash[x[n][i][0]])
        x[n] = newTemp
    return np.array(x)


leafCount = 0
def MTDF(node, guess, depth):
    # print("MTDF COLOR:" + str(node.board.turn))
    # print("MTDF BOARD:" + node.board.__str__())
    # print("DEPTH:" + str(depth))
    upper = 10000
    lower = -10000
    best = guess
    while(lower < upper):
        if guess[0] == lower: 
            beta = guess[0] + 1
        else:
            beta = guess[0]
        guess = minimax(node, beta - 1, beta, depth, False)
        # print(guess[1])
        best = guess
        if guess[0] < beta:
            upper = guess[0]
        else:
            lower = guess[0]
    return best

def iterative_deepening(board, maxdepth, model):
    global engineModel
    global input_details
    global output_details
    engineModel = model
    input_details = engineModel.get_input_details()
    output_details = engineModel.get_output_details()
    guess = [0, None]
    for i in range(maxdepth):
        print(i)
        start = time.time()
        guess = MTDF(board, guess, i + 1)
        print("move=" + str(guess[1]))
        end = time.time()
        print("MTDF took " + str(end - start))
    print(guess)
    return guess[1]

# if search is passed a root node to an evaluated tree
# minimizes and maximizes the tree at every step and returns the best move
# def minimax (node, depth, maxdepth, ismax, alpha, beta, model, color):
def minimax (node, alpha, beta, depth, ismax):
    # return value of terminal nodes
    if (depth == 0 or node.board.is_game_over()):
        global leafCount
        leafCount = leafCount + 1
        prediction = predictFromBoard(node)
        return [prediction, node.board.move_stack[-1]]
    # use recursion to travel to leaf or depth of tree
    # backpropogate max values to root
    # maximize or minimize depending on depth
    global globalBoard
    globalBoard = node
    nextmoves = list(node.board.legal_moves)
    # nextmoves = sorted(nextmoves, key=functools.cmp_to_key(compareBoards))
    if ismax:
        # maximize recursive call
        maxval = -10000
        maxmove = None
        for n in nextmoves :
            node.push(n)
            val = minimax(node, alpha, beta, depth - 1, not ismax)
            if(val[0] > maxval):
                maxval = val[0]
                maxmove = n
            alpha = max(alpha, maxval)
            node.pop()
            if(beta<= alpha): break 
        return [maxval, maxmove]
    else:
        # minimize recursive call
        minval = 10000
        minmove = None
        for n in nextmoves :
            node.push(n)
            val = minimax(node, alpha, beta, depth - 1, not ismax)
            if(val[0] < minval):
                minval = val[0]
                minmove = n
            beta = min(minval, beta)
            node.pop()
            if(beta <= alpha) : break
        return [minval, minmove]
    



    