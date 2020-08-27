# data
import pandas as pd
from stockfish import Stockfish
import chess.pgn
import chess.engine
import numpy as np
import io
engine = chess.engine.SimpleEngine.popen_uci("./stockfish-11-win/Windows/stockfish_20011801_x64_modern")
print(engine.ping)
print("START OF DOCUMENT")

# read the pgn of game data
df = pd.read_csv('../chessEngine/util/pgn_split_1.csv')
df = df[int(len(df)/2):]
myTemp = pd.DataFrame()

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
counte = 0
def expandGame(board):
    global x
    global counte
    global myTemp
    print(counte)
    counte = counte + 1
    x = len(board.move_stack)
    if(x < 2):
        return None
    y = board.copy()
    y.pop()
    myTemp = myTemp.append({'board' : y, "FEN" : y.board_fen()}, ignore_index = True)
    expandGame(y)
def expandWhiteGames(board):
    global df
    global x
    if(x < 2):
        return None
    if(not board.turn): 
        y = board.copy()
        y.pop()
        expandWhiteGames(y)
    x = len(board.move_stack)
    y = board.copy()
    y.pop()
    df = df.append({'board' : y, "FEN" : y.board_fen()}, ignore_index = True)
    expandWhiteGames(y)
# evaluate the board with the stockfish engine imported via python chess
count = 0
def evalBoard(board):
    global count
    print(count)
    count += 1
    k = engine.analyse(board, chess.engine.Limit(time=0.05))['score'].relative.__str__()
    print(k)
    return k
# replace all scores involving forced mates with +/- 10000
def replaceForcedMate(x):
    if x[0] == "#" :
        if x[2] == '0' :
            if x[1] == "+" :
                return +5000
            else :
                return -5000
        else : 
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
def addEvalToDataFrame(r, i):
    # evaluate every board with stockfish and add it to the Dataframe
    r['stockfish_eval'] = r['board'].apply(lambda x : evalBoard(x))
    # remove the boards
    r = r.drop("board", axis=1)
    r.to_csv(f"temp_{i}.csv")
    # replace checkmating values
    r['stockfish_eval'] = r['stockfish_eval'].apply(lambda x : replaceForcedMate(x))
def reverseTurn(board):
    board.turn = not board.turn
    if(board.is_check()):
        return board
    else :
        board.turn = not board.turn
        return board
        
# prep for multiprocessing
from multiprocessing import  Pool
from functools import partial
import numpy as np

def parallelize(data, func, num_of_processes=6):
    data_split = np.array_split(data, num_of_processes)
    print("PARALLELIZING")
    if __name__ == "__main__" :
        pool = Pool(num_of_processes)
        data= pd.concat(pool.map(func, data_split))
        print("FINISHED")
        pool.close()
        print("POOL CLOSED")
        # pool.join()
        # print("POOL JOINED")
        return data

def run_on_subset(func, data_subset):
    print("RUNNING ON SUBSET")
    print(len(data_subset))
    data_subset.apply(func)
    return myTemp

def parallelize_on_rows(data, func, num_of_processes=6):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)

# # extract the exact pgn data
# df['[Event "FICS rated standard game"]'] = df['[Event "FICS rated standard game"]'].apply(lambda x : x if x[:2] == "1." else None)
# df = df.dropna()
# df = df.reset_index()
# df = df.drop('index', axis=1)
# dfArr = np.array_split(df, 12)
# i = 0
# for data in dfArr :
#     data.to_csv(f"pgn_split_{i}.csv")
#     i = i + 1
# create a board for each pgn
df['board'] = df['[Event "FICS rated standard game"]'].apply(lambda x : getBoard(x))
df = df.dropna()
df = df.drop('[Event "FICS rated standard game"]', axis=1)
# add the FEN of each board so board can be reproduced after being saved as csv
df['FEN'] = df['board'].apply(lambda x : x.board_fen())
        
# for each board, call recursive function expandGame, which
# cascades through the previous moves and adds every position
# from that game to the dataframe
df = df.append(parallelize_on_rows(df['board'], expandGame), ignore_index = True)
print("FINISHED PARALLELIZING")
print(len(df))
# df['board'] = df['board'].apply(lambda x : reverseTurn(x))
df.drop("board", axis=1).to_csv("expanded_2020_1_2.csv", index=False)
# extract board pd.Series
# y = df['board']
# x = len(y)
# # evaluate every board with stockfish and add it to the Dataframe
# df['stockfish_eval'] = df['board'].apply(lambda x : evalBoard(x))
# df.to_csv("compiled_2020_mates.csv")
# # replace checkmating values
# df['stockfish_eval'] = df['stockfish_eval'].apply(lambda x : replaceForcedMate(x))
# # remove the boards
# df = df.drop("board", axis=1)
# # save the dataframe as a csv
# df.to_csv("compiled_2020_0.csv")
# df['FEN'] = df['FEN'].apply(lambda x : fenToNPArray(x))