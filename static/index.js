// NOTE: this example uses the chess.js library:
// https://github.com/jhlywa/chess.js

var board = null;
var $board = $('#myBoard');
var $done = $('.done');
var $modal = $('.gameEnd');
var $close = $('.close');
var $message = $('.message');
var $descript = $('.descript')
var squareClass = 'square-55d63';
var squareToHighlight = null;
var colorToHighlight = null;
var game = new Chess();
var move = "";

$close.on('click', (e) => {
    e.stopPropagation();
    $modal.removeClass('selected')
})

$modal.unbind().on('click', (e) => {
    console.log("propogated")
    $modal.removeClass('selected');
})

$message.on('click', (e) => {
    console.log("message clicked")
    e.stopPropagation();
})

$('#startPositionBtn').on('click', async() => {
    console.log("BUTTON CLICKED");
    board.start;
    url = `http://bagelchess.com/reset`;
    // url = `http://127.0.0.1:5000/reset`
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
    if (game.history({ verbose: true })[game.history().length - 1].promotion) {
        promotion = 't'
    } else {
        promotion = 'f'
    }
    // url = `http://www.bagelchess.com/move/${source}/${target}/${game.turn()}/${promotion}`;
    url = `http://127.0.0.1:5000/move/${source}/${target}/${game.turn()}/${promotion}`;
    console.log(url);
    $done.addClass("thinking")
    res = await fetch(url).then(res => {
        return res.text()
    })
    console.log(res)
    game.move(res)
    if (game.game_over()) {
        $modal.addClass('selected');
        if (game.in_draw()) {
            $descript.html("Draw")
        } else if (game.in_checkmate()) {
            $descript.html("Computer won")
        } else {
            $descript.html("Stalemate")
        }
    }
    $done.removeClass("thinking")
    var nextMove = game.history({ verbose: true })[game.history().length - 1]
    $board.find('.' + squareClass).removeClass('highlight-black')
    $board.find('.square-' + nextMove.from).addClass('highlight-black')
    squareToHighlight = nextMove.to
    colorToHighlight = 'black'
    board.position(game.fen())
}

function onDrop(source, target) {
    // see if the move is legal
    var move = game.move({
        from: source,
        to: target,
        promotion: 'q' // NOTE: always promote to a queen for example simplicity
    })

    console.log(source + " " + target);

    // illegal move
    if (move === null) return 'snapback';

    if (!game.game_over()) window.setTimeout(() => { fetchComputerMove(source, target) }, 500)
    else {
        $modal.addClass('selected');
        if (game.in_draw()) {
            $descript.html("Draw")
        } else if (game.in_checkmate()) {
            $descript.html("You won!")
        } else {
            $descript.html("Stalemate")
        }
    }
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
$(window).resize(board.resize)