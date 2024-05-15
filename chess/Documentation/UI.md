### UI
- The UI class implements a command line user interface.

#### Properties
- `ai`: The AI that the user will play against.
- `state`: The GameState object that manages the game logic and the rules.

#### Methods
- `handle_algorithm_selection()`: Handles the selection of the AI algorithm.
- `handle_color_selection()`: Handles the selection of the color.
- `handle_difficulty_selection()`: Handles the selection of the difficulty.
- `print_board()`: Prints the board.
- `start()`: Starts the game.

#### Example
```python
game_repository = ChessRepository()
game_repository.initialize_board()
game_state = GameState(game_repository)
ui = UI(game_state)
ui.start()

# Output: Please select algorithm: minimax, mcts
# Input: mcts
# Output: Please select difficulty: easy, medium, hard
# Input: easy
# Output: Please select color: white, black
# Input: white
# Output: The board after the AI makes a move, and a prompt for the next move

```