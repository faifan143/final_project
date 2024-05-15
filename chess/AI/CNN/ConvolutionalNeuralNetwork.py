import tensorflow as tf
import numpy as np
from AI.CNN.TensorConverter import TensorConverter

class ConvolutionalNeuralNetwork:
    def __init__(self):
        """ Initialize the Convolutional Neural Network """
        # Create the model
        self.input_board = tf.keras.Input(shape=(8, 8, 1), name="board")
        self.input_turn = tf.keras.Input(shape=(1,), name="turn")

        # Board branch
        x = tf.keras.layers.Conv2D(64, 3, activation="relu")(self.input_board)
        x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = tf.keras.layers.Flatten()(x)

        # Turn branch
        y = tf.keras.layers.Dense(64, activation="relu")(self.input_turn)

        # Merge the branches
        concat = tf.keras.layers.concatenate([x, y])

        # Add the fully connected layers
        concat = tf.keras.layers.Dense(64, activation="relu")(concat)
        concat = tf.keras.layers.Dense(64, activation="relu")(concat)
        concat = tf.keras.layers.Dense(64, activation="relu")(concat)
        concat = tf.keras.layers.Dense(64, activation="relu")(concat)
        concat = tf.keras.layers.Dense(64, activation="relu")(concat)
        concat = tf.keras.layers.Dense(64, activation="relu")(concat)
        concat = tf.keras.layers.Dense(64, activation="relu")(concat)
        concat = tf.keras.layers.Dense(64, activation="relu")(concat)

        # Add the output layer
        output = tf.keras.layers.Dense(3, activation="softmax")(concat)
        
        # Create the model
        self.model = tf.keras.Model(inputs=[self.input_board, self.input_turn], outputs=output)
        
        # Compile the model
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    def train(self, path):
        """ Train the model
        :param path: The path to the dataset """
        # Load and preprocess the dataset
        board_tensor, turn_tensor, result_tensor = self.preprocess_data(path)
        self.model.fit([board_tensor, turn_tensor], result_tensor, epochs=128, batch_size=64, validation_split=0.3 , verbose=True)

    def predict(self, fen=None, board_tensor=None, turn_tensor=None):
        """ Predict the result
        :param fen: The FEN of the board
        :param board_tensor: The tensor of the board
        :param turn_tensor: The tensor of the turn
        :return: The prediction of the model
        """
        if fen is not None:
            # Instantiate the converter
            converter = TensorConverter()

            # Convert the data
            board_tensor, turn_tensor = converter.convert_for_prediction(fen)

        return self.model.predict([board_tensor, turn_tensor])

    def save(self, path):
        """ Save the model
        :param path: The path to save the model """
        self.model.save(path)

    def load(self, path):
        """ Load the model
        :param path: The path to the model """
        self.model = tf.keras.models.load_model(path)

    @staticmethod
    def preprocess_data(path):
        """ Preprocess the data
         :param path: The path to the dataset """
        # Instantiate the converter
        converter = TensorConverter()

        # Convert the data
        board_tensor, turn_tensor, result_tensor = converter.convert(path)

        return board_tensor, turn_tensor, result_tensor

if __name__ == "__main__":
    # Create the model
    model = ConvolutionalNeuralNetwork()

    # Train the model
    model.train("Resources/training_dataset.json")
    board_tensor, turn_tensor, result_tensor = model.preprocess_data("Resources/testing_dataset.json")
    # [0, 0, 1] = white wins, [0, 1, 0] = draw, [1, 0, 0] = black wins
    results = model.predict(board_tensor=board_tensor, turn_tensor=turn_tensor)
    model.save("TrainedModels/cnn.h5")
    for result in results:
        if result[0] > result[1] and result[0] > result[2]:
            print("Black wins", round(result[0], 4), round(result[1], 4), round(result[2], 4))
        elif result[1] > result[0] and result[1] > result[2]:
            print("Draw", round(result[0], 4), round(result[1], 4), round(result[2], 4))
        else:
            print("White wins", round(result[0], 4), round(result[1], 4), round(result[2], 4))