import pandas as pd
from stockfish import Stockfish
import chess.pgn
import chess.engine
import numpy as np
import io
engine = chess.engine.SimpleEngine.popen_uci("./stockfish-11-win/Windows/stockfish_20011801_x64_modern")
print(engine.ping)
board = chess.Board("1kr5/ppN2ppr/8/3p2n1/3bb3/8/n3PPPQ/4R1K1 w - - 0 1")
k = engine.analyse(board, chess.engine.Limit(time=10))['score'].relative.__str__()
print(k)