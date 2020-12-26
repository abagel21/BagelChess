// NOTE: this example uses the chess.js library:
// https://github.com/jhlywa/chess.js

var board = null
var $board = $('#myBoard')
var $done = $('.done')
var squareClass = 'square-55d63'
var squareToHighlight = null
var colorToHighlight = null
var game = new Chess()
var move = "";
$('#startPositionBtn').on('click', async() => {
    console.log("BUTTON CLICKED");
    board.start;
    // url = `https://bagelchess.herokuapp.com/reset`;
    url = `http://127.0.0.1:5000/reset`
    res = await fetch(url).then(res => {
        return res.text()
    })
    game.reset();
    board = Chessboard('myBoard', config)
})
$("#status").text("Make a move to begin!");

$('#selfBtn').on('click', selfPlay())

function onDragStart(source, piece, position, orientation) {
    // do not pick up pieces if the game is over
    if (game.game_over()) return false

    // only pick up pieces for White
    if (piece.search(/^b/) !== -1) return false
}

async function fetchComputerMove(source, target) {
    // let fen = game.fen()
    // fen = fen.replace(/\//g, "H")
    // fen = encodeURIComponent(fen)
    console.log(game.turn());
    console.log(game.fen())
        // url = `https://bagelchess.herokuapp.com/move/${source}/${target}/${game.turn()}`;
    url = `http://127.0.0.1:5000/move/${source}/${target}/${game.turn()}`;
    console.log(url);
    $done.addClass("thinking")
    res = await fetch(url).then(res => {
        return res.text()
    })
    game.move(res)
    $done.removeClass("thinking")
    var nextMove = game.history({ verbose: true })[game.history().length - 1]
    $board.find('.' + squareClass).removeClass('highlight-black')
    $board.find('.square-' + nextMove.from).addClass('highlight-black')
    squareToHighlight = nextMove.to
    colorToHighlight = 'black'
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

    console.log(source + " " + target);

    // illegal move
    if (move === null) return 'snapback';

    window.setTimeout(() => { fetchComputerMove(source, target) }, 500)
}

async function selfPlay() {
    // while (!game.game_over()) {
    //     let fen = game.fen()
    //     fen = fen.replace(/\//g, "H")
    //     fen = encodeURIComponent(fen)
    //     url = `http://127.0.0.1:5000/move/${fen}/${game.turn()}`
    //     res = await fetch(url).then(res => {
    //         return res.text()
    //     })
    //     game.move(res)
    //     board.position(game.fen())
    // }
    // alert("Game Over!")
}

// update the board position after the piece snap
// for castling, en passant, pawn promotion
function onSnapEnd() {
    board.position(game.fen())
}

function onMoveEnd() {
    console.log(squareToHighlight + ", " + colorToHighlight);
    $board.find('.square-' + squareToHighlight)
        .addClass('highlight-' + colorToHighlight)
    $("#status").text(game.history()[game.history().length - 1]);
}

var config = {
    draggable: true,
    position: 'start',
    onDragStart: onDragStart,
    onDrop: onDrop,
    onSnapEnd: onSnapEnd,
    onMoveEnd: onMoveEnd
}
board = Chessboard('myBoard', config)