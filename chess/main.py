from flask import Flask, request, jsonify
from Chess.GameState import GameState
from Chess.ChessRepository import ChessRepository
from AI.CNN.ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork
from Chess.IllegalMoveException import IllegalMove
from Chess.WrongColor import WrongColor
from user_interface import UI 
import time

app = Flask(__name__)

chess_repository = ChessRepository()
ui = None
result = None

@app.route('/start_game', methods=['POST'])
def startGame():
    """
        algorithm = {"minimax": "Minimax","mcts": "MCTS"}
        
        difficulty = {"easy": 2,"medium": 5,"hard": 10}

    """
    global chess_repository , ui
    fen_str = request.json['fen']
    algorithm = request.json['algorithm']
    difficulty = request.json['difficulty']
    if difficulty == "" or difficulty == None or algorithm == "" or algorithm == None or algorithm == " " or difficulty == " ":
        return jsonify({
            "error":"true",
            "msg":"json parameters are not completed"
        }) 
    else :
        chess_repository.initialize_board(fen=str(fen_str) if fen_str!="" else "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        game_state = GameState(chess_repository)
        cnn = ConvolutionalNeuralNetwork()
        cnn.load("AI/CNN/TrainedModels/cnn.h5")
        ui = UI(game_state, cnn)
        ui.start(algorithm,int(difficulty))
        ui.print_board(ui.state.board)
        # print(chess.Board(ui.state.fen))
        return jsonify({"error":"false","successful":"true"})

@app.route('/make_move', methods=['POST'])
def make_move():
    start_time = time.time()
    global ui , result

    move = request.json['move']
    
    # try:
    ui.state.make_move(move)
    ui.print_board(ui.state.board)
    print(move)
    print(ui.state.get_value())

    # except IllegalMove as e:
    #     return jsonify({"error":"true","over":"false","msg":str(e)}),400
    # except WrongColor as e:
    #     return jsonify({"error":"true","over":"false","msg":str(e)}),400
    # except IndexError:
    #     return jsonify({"error":"true","over":"false","msg":"index error"}),400
    

    ai_move = ui.ai.select_move(ui.state)
    ui.state.make_move(ai_move)
    ui.print_board(ui.state.board)
    print(ai_move)
    print(ui.state.get_value())

    print(ui.state.get_result())

    end_time = time.time()

    execution_time = end_time - start_time
    
    return jsonify({"error":"false","over":"false","ai_move": ai_move,"board_eval":str(ui.state.get_value()),"time":execution_time})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)