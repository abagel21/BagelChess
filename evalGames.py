# data
import pandas as pd
from stockfish import Stockfish
import chess.pgn
import chess.engine
import numpy as np
import io
import time
import multiprocessing

if __name__ == '__main__':
    engine = chess.engine.SimpleEngine.popen_uci("./stockfish-11-win/Windows/stockfish_20011801_x64_modern")
    print(engine.ping)
    print("START OF DOCUMENT")

    # read the pgn of game data
    df = pd.read_csv('../chessEngine/util/partially_processed_standard_2020/expanded/expanded_2020_3_0.csv')
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
                return 10000 - 10 * int(x[2])
            else :
                return -10000 + 10 * int(x[2])
        else : 
            if x[1] == "+" :
                return +10000
            else:
                return -10000
    else:
        return x
bithash = {
    "." : np.array([0,0,0,0,0,0, 0,0,0,0,0,0]),
    "r" : np.array([0,0,0,1,0,0, 0,0,0,0,0,0]),
    "b" : np.array([0,0,1,0,0,0, 0,0,0,0,0,0]),
    "q" : np.array([0,0,0,0,1,0, 0,0,0,0,0,0]),
    "k" : np.array([0,0,0,0,0,1, 0,0,0,0,0,0]),
    "n" : np.array([0,1,0,0,0,0, 0,0,0,0,0,0]),
    "p" : np.array([1,0,0,0,0,0, 0,0,0,0,0,0]),
    "R" : np.array([0,0,0,0,0,0, 0,0,0,1,0,0]),
    "B" : np.array([0,0,0,0,0,0, 0,0,1,0,0,0]),
    "Q" : np.array([0,0,0,0,0,0, 0,0,0,0,1,0]),
    "K" : np.array([0,0,0,0,0,0, 0,0,0,0,0,1]),
    "N" : np.array([0,0,0,0,0,0, 0,1,0,0,0,0]),
    "P" : np.array([0,0,0,0,0,0, 1,0,0,0,0,0]),
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
    return data_subset.apply(func)

def parallelize_on_rows(data, func, num_of_processes=multiprocessing.cpu_count()):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)

if __name__ == '__main__':
    # extract board pd.Series
    df['board'] = df['FEN'].apply(lambda x : chess.Board(x))
    # df['board'] = df['board'].apply(lambda x : reverseTurn(x))
    y = df['board']
    x = len(y)
    # evaluate every board with stockfish and add it to the Dataframe
    start = time.time()
    df['stockfish_eval'] = parallelize_on_rows(y, evalBoard)
    end = time.time()
    print("Evaluating the boards took " + str((end - start)/60.0) + " minutes")
    # replace checkmating values
    if(df['stockfish_eval'][0] is not None) :
        print(df.head(5))
        df.drop("board", axis=1).to_csv("compiled_2020_mates_3_0.csv", index=False)
        df['stockfish_eval'] = df['stockfish_eval'].apply(lambda x : replaceForcedMate(x))
        # remove the boards
        df = df.drop("board", axis=1)
        #   save the dataframe as a csv
        df.to_csv("compiled_2020_3_0.csv", index=False)