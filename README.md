# BagelChess
BagelChess is a deep learning chess engine. Utilizing a combination of minimax with alpha beta pruning and a convolutional neural network, the engine seeks to select the most advantageous move from any given position.

## Planned Development
Short Term :
- Improve model structure and hyperparameters
- Optimize current move selection strategy

Long Term :
- Train a larger dataset
- Improve move selection with optimization of the move tree, or with Monte Carlo Tree Search
- Build my own bitboard game representation and move generator

## Construction
As my first foray into chess programming and an early foray into machine learning, I sought to pick a structure that would be challenging to implement and require research but not be unnattainable. As such, I decided to work with minimax instead of a Monte Carlo Tree search, and a simpler goal for my neural network-- evaluating positions from a board representation without worrying about move probabilities, focused instead on pure centipawn rating.

I obtained my data from the fics database, starting with a smaller sample of 5000 games, expanded to all positions that were present during the game to create an 800,000 sample dataframe. I then used stockfish to produce evaluations for every position, creating the validation data for supervised learning.

After a lot of tweaking to avoid a severe overfitting problem, my neural network looked like this:
// image will go here

I used dropout layers, regularization, batch normalization, and a simpler layout to avoid the overfitting issue. My final model achieved a mean absolute error of x, which is pretty good for a scale from -10000 to 10000. 

I also constructed a website for the engine where it can be played against utilizing flask and two chess libraries--one for the board, one for the game--using libraries because I had limited time and my focus was on the engine side.

## Dependencies
- Python-Chess
- Chess.js
- Chessboard.js
- Tensorflow
- Keras
- Numpy
- Pandas

# Get Started
Download the package and unzip it. A working model is included as well as all the data preprocessing and utility classes for minimax and the model. By running app.py, you can work with a fully functioning hosted chess game against my engine, or you can process a new model using my data preprocessing and training functions by finding any pgn and converting it to a csv.
