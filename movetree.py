import chess
import chess.engine
import numpy as np
from stockfish import Stockfish
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
def fenToNPArray(x):
    x = chess.Board(x)
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
class Node :
    def __init__(self, board, model, color) :
        # print("INIT CALLED")
        self.board = board
        # if(not board.turn) :  self.board = board.mirror()
        print(board.fen())
        print(color)
        x = np.stack(np.array(fenToNPArray(board.fen())))
        x = x.reshape(1,8,8,6)
        # predict the value of the board state with the CNN
        self.value = model.predict(x)[0]
        print(self.value)
        # find all moves for White/Black (depending on color)
        self.nextMoves = board.legal_moves