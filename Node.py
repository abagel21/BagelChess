from zobristHashing import *
# a dynamic node class designed to store the updated zobrist hash of a chess.Board object
class Node:
    def __init__(self, board):
        self.hasher = Hasher()
        self.board = board.copy()
        self.pos = self.hasher.get_pos_hash(self.board)
        self.pos_stack = [self.pos]
        self.zobrist = self.hasher.update_hash(self.board, self.pos)
        self.zobrist_stack = [self.zobrist]
    def push(self, move):
        pos_change = self.hasher.updated_position_hash(board=self.board, move=move)
        self.pos ^= pos_change
        self.pos_stack.append(self.pos)
        self.board.push(move)
        self.zobrist = self.hasher.update_hash(self.board, self.pos)
        self.zobrist_stack.append(self.zobrist)
    def pop(self):
        self.board.pop()
        self.pos_stack.pop()
        self.pos=self.pos_stack[-1]
        self.zobrist_stack.pop()
        self.zobrist=self.zobrist_stack[-1]