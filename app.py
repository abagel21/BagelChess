from flask import Flask, render_template
from flask_cors import CORS
from flask import send_from_directory
import os
import chess
import chess.pgn
import re
from tensorflow.keras.models import load_model
from treesearch import minimaxRoot
from movetree import Node
import urllib.parse
import tensorflow as tf
import tensorflow_addons as tfa
tf.keras.optimizers.AdamW = tfa.optimizers.AdamW

DEPTH = 4
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], True)
model = load_model('./saved_models/optimal')
converter = tf.lite.TFLiteConverter.from_saved_model("./saved_models/optimal")
model = converter.convert()
model = tf.lite.Interpreter(model_content=model)
model.allocate_tensors()
PORT = 80

app = Flask(__name__)
CORS(app)
board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
print(str(minimaxRoot(board, 0, 2, False, model, True)))

@app.route('/')
def home():
    # serve static
    global board
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    print(board.copy().piece_map)
    return render_template('index.html')
    # return "HELLO"

@app.route('/move/<string:source>/<string:target>/<string:color>') 
def move(source, target, color):
    color = False if color == 'w' else True
    # need color of engine and FEN of current board
    # if(board.is_checkmate):
    #     return "CHECKMATE"
    board.turn = color
    prevMove = chess.Move(chess.SQUARE_NAMES.index(source), chess.SQUARE_NAMES.index(target))
    board.push(prevMove)
    for i in range(1, DEPTH + 1):
        move = minimaxRoot(board, 0, i, False, model, not color)
    movestring = board.san(move)
    board.push(move)
    return movestring

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')