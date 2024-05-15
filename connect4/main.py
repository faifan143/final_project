from flask import Flask, request, jsonify
import numpy as np
print(np.__version__)

import torch
print(torch.__version__)
import torch.nn.functional as F
torch.manual_seed(0)

from connect4 import ConnectFour
from resnet import ResNet
from mcts import MCTS

     
app = Flask(__name__)

game = None
mcts = None
state = None

@app.route('/start', methods=['POST'])
def strat_game():
    global game , mcts , state
    game = ConnectFour()
    """    
        args = {
            'C': 2,
            'num_searches': 600,
            'dirichlet_epsilon': 0.0,
            'dirichlet_alpha': 0.3
        }
    """
    args = request.json['args']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet(game, 9, 128, device)
    model.load_state_dict(torch.load("model_7_ConnectFour.pt", map_location=device))
    model.eval()

    mcts = MCTS(game, args, model)

    state = game.get_initial_state()
    print(state)
    print(args)
    print("\n")
    return jsonify({'message': 'Game initialized successfully.'})





@app.route('/move', methods=['POST'])
def make_move():
    global state,mcts,game
    action = request.json['move']
    valid_moves = game.get_valid_moves(state)
    if valid_moves[int(action)] == 0:
        return jsonify({
            'finished':'false',
            'error': 'Invalid action. Please choose a valid action.'
            }), 400
    state = game.get_next_state(state, int(action), 1)
    print(state)



    neutral_state = game.change_perspective(state, -1)
    mcts_probs = mcts.search(neutral_state)
    ai_action = np.argmax(mcts_probs)
    state = game.get_next_state(state, ai_action, -1)
    print(state)
    new_value, new_is_terminal = game.get_value_and_terminated(state, ai_action)

    
    return jsonify({"state": int(ai_action)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)