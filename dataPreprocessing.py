# data
import pandas as pd
from stockfish import Stockfish
import chess.pgn
import chess.engine
import numpy as np
import io
engine = chess.engine.SimpleEngine.popen_uci("./stockfish-11-win/Windows/stockfish_20011801_x64")
fishy = Stockfish('./stockfish-11-win/Windows/stockfish_20011801_x64')

# read the pgn of game data
df = pd.read_csv('../chessEngine/ficsgamesdb_search_146961.csv')
# extract the exact pgn data
df['moveset'] = df['moveset'].apply(lambda x : x if x[:2] == "1." else None)
df = df.dropna()
df = df.reset_index()
df = df.drop('index', axis=1)
# create a board for each pgn
df['board'] = df['moveset'].apply(lambda x : getBoard(x))
df = df.dropna()
df = df.drop("moveset", axis=1)
# add the FEN of each board so board can be reproduced after being saved as csv
df['FEN'] = df['board'].apply(lambda x : x.board_fen())

# get the board from the PGN representation
def getBoard(pgnPos) :
    pgnPos = pgnPos[:len(pgnPos)]
    pgnStr = io.StringIO(pgnPos)
    pgnGame = chess.pgn.read_game(pgnStr)
    board = pgnGame.board()
    for move in pgnGame.mainline_moves():
        board.push(move)
    if(len(board.move_stack) == 0):
        return None
    return board
# get the FEN representation of the board for Stockfish evaluation
def getFen(board):
    return board.board_fen()
# traceback the moves for each game and append them to the dataframe
def expandGame(board):
    global df
    x = len(board.move_stack)
    if(x < 2):
        return None
    y = board.copy()
    y.pop()
    df = df.append({'board' : y, "FEN" : y.board_fen()}, ignore_index = True)
    expandGame(y)
# evaluate a board with the stockfish library
def evalGame(board):
    print(len(board.move_stack))
    fen = getFen(board)
    print(fen)
    fishy.set_fen_position(fen)
    print(fishy.get_board_visual())
    r = fishy.get_evaluation()
    print(r)
    return r['value']
# evaluate the board with the stockfish engine imported via python chess
def evalBoard(board):
    print(len(board.move_stack))
    return engine.analyse(board, chess.engine.Limit(time=0.3))['score'].relative.__str__()
# replace all scores involving forced mates with +/- 10000
def replaceForcedMate(x):
    if x[0] == "#" :
       if x[1] == "+" :
           return +10000
       else:
           return -10000
    else:
       return x
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
def boardToString(x):
    x = chess.Board(x)
    x=x.__str__()
    return x
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
# extract board pd.Series
y = df['board']
x= len(y)
# for each board, call recursive function expandGame, which
# cascades through the previous moves and adds every position
# from that game to the dataframe
for i in range(x+1):
    print(i)
    expandGame(y[i])
# evaluate every board with stockfish and add it to the Dataframe
df['stockfish_eval'] = df['board'].apply(lambda x : evalBoard(x))
# replace checkmating values
df['stockfish_eval'] = df['stockfish_eval'].apply(lambda x : replaceForcedMate(x))
# remove the boards
df = df.drop("board", axis=1)
# save the dataframe as a csv
df.to_csv("compiled_chess_games.csv")
df['FEN'] = df['FEN'].apply(lambda x : fenToNPArray(x))