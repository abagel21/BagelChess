// NOTE: this example uses the chess.js library:
// https://github.com/jhlywa/chess.js

var board = null
var game = new Chess()
$('#startPositionBtn').on('click', () => {
    console.log("BUTTON CLICKED");
    board.start;
    game.reset();
})

function onDragStart(source, piece, position, orientation) {
    // do not pick up pieces if the game is over
    if (game.game_over()) return false

    // only pick up pieces for White
    if (piece.search(/^b/) !== -1) return false
}

async function fetchComputerMove() {
    let fen = game.fen()
    fen = fen.replace(/\//g, "H")
    fen = encodeURIComponent(fen)
    url = `http://127.0.0.1:5000/move/${fen}/${game.turn()}`
    res = await fetch(url).then(res => {
        return res.text()
    })
    game.move(res)
    board.position(game.fen())
}

function onDrop(source, target) {
    if (game.game_over()) {
        alert("Game Over!")
    }
    // see if the move is legal
    var move = game.move({
        from: source,
        to: target,
        promotion: 'q' // NOTE: always promote to a queen for example simplicity
    })

    // illegal move
    if (move === null) return 'snapback'

    // make random legal move for black
    window.setTimeout(fetchComputerMove, 500)
}

// update the board position after the piece snap
// for castling, en passant, pawn promotion
function onSnapEnd() {
    board.position(game.fen())
}

var config = {
    draggable: true,
    position: 'start',
    onDragStart: onDragStart,
    onDrop: onDrop,
    onSnapEnd: onSnapEnd
}
board = Chessboard('myBoard', config)