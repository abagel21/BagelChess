from flask import Flask, render_template, session
from flask_session import Session
from flask_cors import CORS, cross_origin
from flask import send_from_directory
import os
import chess
import chess.pgn
import re
from tensorflow.keras.models import load_model
from treesearch import minimaxRoot
from mtdf import iterative_deepening
import urllib.parse
import tensorflow as tf
import tensorflow_addons as tfa
from flask_pymongo import PyMongo
from Node import Node
tf.keras.optimizers.AdamW = tfa.optimizers.AdamW

DEPTH = 3
# model = load_model('optimal')
model = tf.lite.Interpreter("model.tflite")
model.allocate_tensors()
PORT = 80

application = Flask(__name__)
CORS(application)
board = None
board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

@application.route('/')
def home():
    print("HOME ROUTE HIT")
    # serve static
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    print(board)
    return render_template('index.html')

@application.route('/move/<string:source>/<string:target>/<string:color>/<string:promotion>') 
def move(source, target, color, promotion):
    global board
    localboard = board
    # print("board here")
    # print(localboard)
    # print(localboard.is_game_over())
    # print(color)
    color = False if color == 'w' else True
    promotion = True if promotion == 't' else False
    # print(color)
    # need color of engine and FEN of current board
    # if(localboard.is_game_over(claim_draw=False)):
    #     return chess.Move.null
    localboard.turn = color
    if promotion:
        prevMove = chess.Move(chess.SQUARE_NAMES.index(source), chess.SQUARE_NAMES.index(target), promotion = 5)
    else:
        prevMove = chess.Move(chess.SQUARE_NAMES.index(source), chess.SQUARE_NAMES.index(target))
    localboard.push(prevMove)
    # for i in range(1, DEPTH + 1):
    #     move = minimaxRoot(board, 0, i, False, model, not color)
    # move = minimaxRoot(Node(localboard), 0, DEPTH, False, model, not color)
    # oldMove = move
    move = iterative_deepening(Node(localboard.copy()), DEPTH, model)
    # print(oldMove)
    # print(localboard)
    # print(localboard.turn)
    print(move)
    print(localboard.san(move))
    if(move.promotion):
        move.promotion = 5
    movestring = localboard.san(move)
    localboard.push(move)
    session['board'] = localboard.fen()
    print("MOVE:" + str(move))
    return movestring

@application.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(application.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@application.route('/reset')
def reset():
    print("RESETTING")
    session['board'] = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    return "yes"

@application.route('/session/')
def updating_session():
    res = str(session.items())
    return res