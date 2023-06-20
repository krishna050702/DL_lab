import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import time

sns.set_style('darkgrid')

colors = ["#9b5de5", "#f15bb5", "#fee440", "#00bbf9", "#00f5d4"]


class NeuralNetwork:
    """
    A class to represent a neural network.
    """

    def __init__(self):
        """
        Constructs all the necessary attributes for the neural network object.
        """

        # Load the Sonar dataset
        self.X, self.y = make_classification(n_samples=208, n_features=60, n_informative=60, n_redundant=0, n_classes=2,
                                             random_state=42)
        # Split the dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=42)

    def SimpleSequentialModel(self):
        nn = tf.keras.models.Sequential([
            tf.keras.layers.Dense(32, input_dim=60, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        nn._name = "SimpleSequentialModel"

        # Compile the model
        nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # show the model summary
        nn.summary()

        # Train the model and time it
        start = time.perf_counter()
        history = nn.fit(self.X_train, self.y_train, validation_split=0.1, epochs=200, verbose=0)
        end = time.perf_counter()

        # Evaluate the model
        loss, accuracy = nn.evaluate(self.X_test, self.y_test, verbose=0)
        print(f'Testing Accuracy: {round(accuracy * 100, 2)}%, Testing Loss: {round(loss, 2)}')

        # Plot the loss and accuracy of training and validation
        plt.figure(figsize=(12, 8))

        plt.subplot(221).set_title('Training Loss')
        plt.plot(history.history['loss'], label='train loss', color=colors[0])

        plt.subplot(222).set_title('Training Accuracy')
        plt.plot(history.history['accuracy'], label='train accuracy', color=colors[1])

        plt.subplot(223).set_title('Validation Loss')
        plt.plot(history.history['val_loss'], label='val loss', color=colors[2])

        plt.subplot(224).set_title('Validation Accuracy')
        plt.plot(history.history['val_accuracy'], label='val accuracy', color=colors[3])

        plt.suptitle(nn._name)
        plt.show()

        print(f"time : {end - start}s")