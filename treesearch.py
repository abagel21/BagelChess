import math
from movetree import Node

# board will be mirrored if computer is playing black
# so minimax always works, as neural network returns 
# positive for white winning
# and negative for black winning
count = 0
def minimaxRoot(node, depth, maxdepth, ismax, model, color) :
    bestMoveVal = 10000
    bestMove = None
    # build node for each possible move from starting state
    # (which also evaluates the move with the convolutional neural network)
    for n in node.nextMoves :
        newBoard = node.board.copy()
        newBoard.push(n)
        newNode = Node(newBoard, model, color)
        temp = minimax(newNode, depth + 1, maxdepth, True, -10000, 10000, model, not color)
        if( temp <= bestMoveVal) :
            bestMoveVal = temp
            bestMove = n
    return bestMove

# if search is passed a root node to an evaluated tree
# minimizes and maximizes the tree at every step and returns the best move
def minimax (node, depth, maxdepth, ismax, alpha, beta, model, color):
    global count
    count +=1
    print(count)
    # return value of terminal nodes
    if (depth >= maxdepth):
        print("RETURNING")
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
            newNode = Node(newBoard, model, color)
            maxval = max(minimax(newNode, depth + 1, maxdepth, False, alpha, beta, model, not color), maxval)
            alpha = max(alpha, maxval)
            if(beta<= alpha): return maxval
        return maxval
    else:
        # minimize recursive call
        minval = 10000
        for n in node.nextMoves :
            newBoard = node.board.copy()
            newBoard.push(n)
            newNode = Node(newBoard, model, color)
            minval = min(minimax(newNode, depth + 1, maxdepth, True, alpha, beta, model, not color), minval)
            beta = min(minval, beta)
            if(beta <= alpha) : return minval
        return minval
    