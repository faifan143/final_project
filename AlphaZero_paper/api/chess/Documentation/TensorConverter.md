### TensorConverter
- This class is used to convert the dataset into tensors that can be used by the neural network.

#### Properties
- piece_mapping: A dictionary that maps the pieces to integers.
- turn_mapping: A dictionary that maps the turns to integers.
- result_mapping: A dictionary that maps the outcomes of the games to integers.
- fen_piece_mapping: A dictionary that maps the FEN notation of the pieces to the pieces.

#### Methods
- convert_board_from_fen: Converts the board from FEN notation to a 2D array.
- load_and_convert_dataset: Loads the dataset from a JSON file and converts the boards to 2D arrays.
- convert: Loads a dataset of games and creates the tensors of the boards, turns and outcome labels.
- convert_for_prediction: Converts a board from FEN notation to a tensor for prediction.

#### Example
```python
# Create an instance of the TensorConverter class
converter = TensorConverter()

# Convert a gamestate to a tensor
converter.convert_for_prediction("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

# Output: 
# (
#   array(
#       [[[ 1,  2,  3,  4,  5,  3,  2,  1],
#       [ 6,  6,  6,  6,  6,  6,  6,  6],
#       [ 0,  0,  0,  0,  0,  0,  0,  0],
#       [ 0,  0,  0,  0,  0,  0,  0,  0],
#       [ 0,  0,  0,  0,  0,  0,  0,  0],
#       [ 0,  0,  0,  0,  0,  0,  0,  0],
#       [12, 12, 12, 12, 12, 12, 12, 12],
#       [ 7,  8,  9, 10, 11,  9,  8,  7]]]),
#   array([1])
# )
```
