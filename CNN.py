import tensorflow as tf
import pandas as pd
import chess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import datetime

# construct a convolutional neural network intended to evaluate board state inputs as winning, losing, or draw
# train it with FICS game database labelled by stockfish? (unsure of training data for now)
# and test it on a specific game by having 
# it evaluate a board states at each level of a move tree
# to a certain depth 
# and then pass that tree to minimax to find the best move

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], True)

# import preprocessed data
# tf.debugging.set_log_device_placement(True)
df = pd.read_csv("./util/compiled_chess_games_0.csv")
print(len(df))
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
df['FEN'] = df['FEN'].apply(lambda x : fenToNPArray(x))
print(len(df))
X = np.stack(df['FEN'].values)
y = df['stockfish_eval'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# model configuration
# takes in a board input and outputs a centipawn value
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.regularizers import L1, L2
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.initializers import GlorotUniform
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose = 1, patience = 1000)
adamOpti = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam')
initializer = GlorotUniform()
regularizerl2 = L2(l2 = 0.1)
regularizerl1 = L1(l1 = 0.1)
EPOCHS = 500

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization
model = Sequential()
#, kernel_regularizer=regularizerl2
model.add(Conv2D(36, (3, 3), activation='relu', input_shape=(8,8,6), padding='same', kernel_regularizer=regularizerl2))
model.add(Conv2D(36, (3, 3), activation='relu', padding='same', data_format="channels_last", kernel_regularizer=regularizerl2))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))


model.add(Flatten())
model.add(Dense(64, activation = 'relu', kernel_initializer=initializer, kernel_regularizer=regularizerl2))
model.add(Dropout(0.5))
model.add(Dense(1))
# , metrics=[MeanAbsoluteError(name='mean_absolute_error', dtype=None)]
model.compile(loss='mse', optimizer=adamOpti)

# training
# , callbacks=[early_stop]
model.fit(X_train, y_train, epochs = EPOCHS, validation_data = (X_test, y_test), batch_size=1024, callbacks=[tensorboard_callback, early_stop])

# evaluating
from sklearn.metrics import mean_absolute_error, explained_variance_score
pred = model.predict(X_test)
print(mean_absolute_error(y_test, pred))
print(explained_variance_score(y_test, pred))
# to see logs
# tensorboard --logdir logs/fit



# saving
# model_no = 3
# model.save(f'saved_models/chess_model_{model_no}')