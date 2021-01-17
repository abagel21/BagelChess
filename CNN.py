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

# construct a convolutional neural network intended to evaluate board state inputs as winning, losing, or draw
# train it with FICS game database labelled by stockfish? (unsure of training data for now)
# and test it on a specific game by having 
# it evaluate a board states at each level of a move tree
# to a certain depth 
# and then pass that tree to minimax to find the best move
# need to store pieces (obviously) turn, en passant square, castling rights for four directions
if __name__ == '__main__':
    print(multiprocessing.cpu_count())
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("WAS POPPIN")
    # import preprocessed data
    # tf.debugging.set_log_device_placement(True)
    df = pd.read_csv("compiled_2020_0-3.csv")
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
    print("TOOK " + str(end - start))
    print(len(df))
    print(df.head())
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
    from tensorflow_addons.optimizers import AdamW
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose = 1, patience = 15)
    adamOpti = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam')
    decayedAdamOpti = AdamW(weight_decay = 0.00001, learning_rate = 0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='AdamW')
    sgdOpti = SGD(
        learning_rate=0.0001, momentum=0.8, nesterov=False, name='SGD'
    ) 
    rmsOpti = tf.keras.optimizers.RMSprop(
        learning_rate=0.001, rho=0.9, momentum=0.8, epsilon=1e-07, centered=False,
        name='RMSprop'
    )
    initializer = GlorotUniform()
    regularizerl2 = L2(l2 = 0.1)
    EPOCHS = 3000

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization
    model = Sequential()
    #, kernel_regularizer=regularizerl2
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(8,8,11), padding='same', kernel_regularizer=regularizerl2))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(8,8,11), padding='same', kernel_regularizer=regularizerl2))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(8,8,11), padding='same', kernel_regularizer=regularizerl2))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(8,8,11), padding='same', kernel_regularizer=regularizerl2))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(8,8,11), padding='same', kernel_regularizer=regularizerl2))
    model.add(BatchNormalization())
    # model.add(MaxPooling2D((2,2)))
    # model.add(Dropout(0.2))


    model.add(Flatten())
    model.add(Dense(64, activation = 'relu', kernel_initializer=initializer, kernel_regularizer=regularizerl2))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation = 'relu', kernel_initializer=initializer, kernel_regularizer=regularizerl2))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    # , metrics=[MeanAbsoluteError(name='mean_absolute_error', dtype=None)]
    model.compile(loss='mse', optimizer=decayedAdamOpti)

    # training
    # , callbacks=[early_stop]
    modelStart = time.time()
    model.fit(X_train, y_train, epochs = EPOCHS, validation_data = (X_test, y_test), batch_size=1024, callbacks=[tensorboard_callback, early_stop])
    modelEnd = time.time()

    # evaluating
    from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
    pred = model.predict(X_test)
    print("Validation Scores")
    print("MSE=" +  str(mean_squared_error(y_test, pred)))
    print("MAE=" + str(mean_absolute_error(y_test, pred)))
    print("Explained Variance=" + str(explained_variance_score(y_test, pred)))
    print("Converting from FEN to NP array took " + str((end - start)/60.0) + " minutes")
    print("Training model took " + str((modelEnd - modelStart)/60.0) + " minutes")
    # to see logs
    # tensorboard --logdir logs/fit



    # saving
    model_no = "standard_june_12"
    model.save(model_no)