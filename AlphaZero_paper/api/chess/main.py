from flask import Flask, request, jsonify
from Chess.GameState import GameState
from Chess.ChessRepository import ChessRepository
from AI.CNN.ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork
from Chess.IllegalMoveException import IllegalMove
from Chess.WrongColor import WrongColor
from user_interface import UI 


app = Flask(__name__)


chess_repository = ChessRepository()
chess_repository.initialize_board()
game_state = GameState(chess_repository)
cnn = ConvolutionalNeuralNetwork()
cnn.load("AI/CNN/TrainedModels/cnn.h5")
ui = UI(game_state, cnn)
ui.start()
result = None

@app.route('/make_move', methods=['POST'])
def make_move():
    global ui , result
    move = request.json['move']
    try:
        ui.state.make_move(move)
        ui.print_board(ui.state.board)
    except IllegalMove as e:
        return jsonify({
            "error":"true",
            "over":"false",
            "msg":e})
    except WrongColor as e:
        return jsonify({
            "error":"true",
            "over":"false",
            "msg":e})
    except IndexError:
        return jsonify({
            "error":"true",
            "over":"false",
            "msg":"index error"})
    if ui.state.board.game_over :
        result = ui.state.get_result()
        return jsonify({
            "error":"false",
            "over":"true",
            "result": result})


    ai_move = ui.ai.select_move(ui.state)
    ui.state.make_move(ai_move)
    ui.print_board(ui.state.board)
    if ui.state.board.game_over :
        result = ui.state.get_result()
        return jsonify({
            "error":"false",
            "over":"true",
            "result": result})


    return jsonify({
        "error":"false",
        "over":"false",
        "ai_move": ai_move})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)