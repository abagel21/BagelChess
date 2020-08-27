import tensorflow as tf
import pandas as pd
import chess
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
tf.keras.optimizers.AdamW = tfa.optimizers.AdamW
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, explained_variance_score
test = pd.read_csv("./util/compiled_chess_games_0.csv")
model = load_model('./saved_models/optimal')
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
test['FEN'] = test['FEN'].apply(lambda x : fenToNPArray(x))
X_final = np.stack(test['FEN'].values)
y_final = test['stockfish_eval'].values
pred = model.predict(X_final)
print("Test Scores")
print(mean_absolute_error(y_final, pred))
print(explained_variance_score(y_final, pred))