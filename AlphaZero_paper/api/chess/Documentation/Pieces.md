### Piece
- Base class for all chess pieces.

#### Properties
- `type`: The type of the piece. It can be either `pawn`, `rook`, `knight`, `bishop`, `queen` or `king`.
- `color`: The color of the piece. It can be either `w` or `b`.
- `position`: A tuple that indicates the position of the piece on the board. The first element of the tuple is the row and the second element is the column.

#### Methods
- `get_legal_moves(board, move_history, pieces)`: Returns a list of all the legal moves for the piece.
- `get_value()`: Returns the value of the piece.
- `__repr__()`: Returns a string representation of the piece.

