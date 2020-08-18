import math
from movetree import Node

# positive for white winning
# and negative for black winning
def minimaxRoot(node, depth, maxdepth, ismax, model, color) :
    bestMoveVal = 10000
    bestMove = None
    # build node for each possible move from starting state
    # (which also evaluates the move with the convolutional neural network)
    for n in node.nextMoves :
        newBoard = node.board.copy()
        newBoard.push(n)
        newNode = Node(newBoard, model, color, False)
        temp = minimax(newNode, depth + 1, maxdepth, True, -10000, 10000, model, not color)
        if( temp <= bestMoveVal) :
            bestMoveVal = temp
            bestMove = n
    return bestMove

# if search is passed a root node to an evaluated tree
# minimizes and maximizes the tree at every step and returns the best move
def minimax (node, depth, maxdepth, ismax, alpha, beta, model, color):
    # return value of terminal nodes
    if (depth >= maxdepth):
        return node.value
    # use recursion to travel to leaf or depth of tree
    # backpropogate max values to root
    # maximize or minimize depending on depth
    if ismax:
        # maximize recursive call
        maxval = -10000
        for n in node.nextMoves :
            newBoard = node.board.copy()
            newBoard.push(n)
            newNode = Node(newBoard, model, color, True if depth == maxdepth - 1 else False)
            maxval = max(minimax(newNode, depth + 1, maxdepth, False, alpha, beta, model, not color), maxval)
            alpha = max(alpha, maxval)
            if(beta<= alpha): break
        return maxval
    else:
        # minimize recursive call
        minval = 10000
        for n in node.nextMoves :
            newBoard = node.board.copy()
            newBoard.push(n)
            newNode = Node(newBoard, model, color, True if depth == maxdepth - 1 else False)
            minval = min(minimax(newNode, depth + 1, maxdepth, True, alpha, beta, model, not color), minval)
            beta = min(minval, beta)
            if(beta <= alpha) : break
        return minval
    