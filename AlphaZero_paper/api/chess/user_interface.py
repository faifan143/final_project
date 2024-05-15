from AI.MCTS.monte_carlo_tree_search import MCTS
from AI.Minimax.minimax import Minimax

class UI:
    def __init__(ui, game_state, cnn):
        """ Initializes the UI
        :param game_state: The game state to use """
        ui.ai = None
        ui.state = game_state
        ui.cnn = cnn

    def handle_algorithm_selection(ui):
        """ Handles the selection of the AI algorithm """
        commands = {"minimax": "Minimax",
                    "mcts": "MCTS"
                    }

        while True:
            algorithm = input("Please select algorithm: minimax, mcts\n> ")
            if algorithm in commands:
                return commands[algorithm]
            else:
                print("Invalid algorithm")

    def handle_color_selection(ui):
        """ Handles the selection of color the AI will be playing """
        commands = {"white": "w",
                    "black": "b"
                    }

        while True:
            color = input("Please select color: white, black\n> ")
            if color in commands:
                return commands[color]
            else:
                print("Invalid color")

    def handle_difficulty_selection(ui):
        """ Handles the difficulty that the AI will be playing at """
        commands = {"easy": 2,
                    "medium": 5,
                    "hard": 10
                    }

        while True:
            difficulty = input("Please select difficulty: easy, medium, hard\n> ")
            if difficulty in commands:
                return commands[difficulty]
            else:
                print("Invalid difficulty")

    def print_board(ui, board):
        """ Print the board

        :param board: The board to print"""
        for i in range(len(board.board) - 1, -1, -1):
            string = []
            for j in range(len(board.board[i])):
                if board.board[i][j] is not None:
                    string.append(board.board[i][j])
                else:
                    string.append("")
            print(string)
        print("\n")


    def start(self):
        """ Starts the game """
        algorithm = self.handle_algorithm_selection()
        difficulty = self.handle_difficulty_selection()
        color = "b" # self.handle_color_selection()
        if algorithm == "MCTS":
            self.ai = MCTS(self.state, iterations=difficulty, depth_limit=None, use_opening_book=True, cnn=self.cnn)
        else:
            self.ai = Minimax(self.state, difficulty, color)
