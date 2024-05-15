### Minimax
- This class implements the Minimax algorithm. The algorithm has alpha-beta pruning implemented to speed up the search.

#### Properties
- `state`: The GameState object that manages the game logic and the rules.
- `depth`: The depth limit of the tree.
- `hashtable`: An object of the HashTable class that is used to store the nodes of the tree.
- `color`: The color of the player. It is 1 if the player is white and -1 if the player is black. Used to determine the best move.

#### Methods
- `min_value(state, depth, alpha, beta)`: Returns the minimum value of the given state.
- `max_value(state, depth, alpha, beta)`: Returns the maximum value of the given state.
- `select_move(state)`: Performs the Minimax algorithm and selects the best move.