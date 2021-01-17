import tensorflow as tf
import pandas as pd
import chess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import multiprocessing
import tensorflow as tf
import datetime
import time

if __name__ == '__main__':
    print(multiprocessing.cpu_count())
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("WAS POPPIN")
    # import preprocessed data
    # tf.debugging.set_log_device_placement(True)
    df = pd.read_csv("compiled_2020_0-3.csv")[6000000:]
    print(len(df))
    print(df.head())
    print("HELLLLLLLLLLLLOOOOOOOOOOOOOOOOO")
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
# converts the FEN representation of a board into an 8x8 array of squares of size 7
# the first six encode {-1,0,1} for black or white pieces. 7th includes turn to move. 8-11 include castling rights
counte = 0
def fenToNPArray(x):
    global counte
    x = chess.Board(x)
    board = x
    c = x.turn
    counte = counte + 1
    if __name__ == '__main__':
        print(counte)
    if x.castling_rights != 0:
        print(x.castling_rights)
    x=x.__str__()
    x = x.split("\n")
    for n in range(len(x)):
        x[n] = np.array(x[n].split()).reshape(8,1)
        newTemp = []
        for i in range(len(x[n])) :
            # add pieces and turn
            temp = np.append(bithash[x[n][i][0]], 1 if c else -1)
            # add castling rights
            temp = np.append(temp, board.has_kingside_castling_rights(True))
            temp = np.append(temp, board.has_queenside_castling_rights(True))
            temp = np.append(temp, board.has_kingside_castling_rights(False))
            temp = np.append(temp, board.has_queenside_castling_rights(False))
            # add en passant?
            newTemp.append(temp)
        x[n] = newTemp
    return np.array(x)
from multiprocessing import  Pool
if __name__ == '__main__':
    start = time.time()
    with Pool(multiprocessing.cpu_count()) as p:
        df['FEN'] = list(p.imap(fenToNPArray, df['FEN'], 6))
    # df['FEN'] = df['FEN'].apply(lambda x : fenToNPArray(x))
    end = time.time()
    df.to_pickle("currentCompiled_1.csv")
    print("TOOK " + str(end - start))
    print(len(df))
    print(df.head())