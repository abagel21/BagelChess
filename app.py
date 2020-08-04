from flask import Flask, render_template
from flask_cors import CORS
import os
import chess
import re
from tensorflow.keras.models import load_model
from treesearch import minimaxRoot
from movetree import Node
import urllib.parse
import tensorflow as tf

DEPTH = 2
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], True)
model = load_model('./saved_models/3.1368e6')
PORT = 80

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    # serve static
    return render_template('index.html')

@app.route('/move/<string:fen>/<string:color>') 
def move(fen, color):
    print("Endpoint hit")
    fen = urllib.parse.unquote(fen)
    fen = re.sub(r'H', '/', fen)
    print(fen)
    color = True if color == 'w' else False
    # need color of engine and FEN of current board
    board = chess.Board(fen)
    board.turn = color
    node = Node(board, model, color)
    move = minimaxRoot(node, 0, DEPTH, False, model, not color)
    print(board.san(move))
    move = board.san(move)
    return move

if __name__ == "__main__":
    app.run(debug=True)