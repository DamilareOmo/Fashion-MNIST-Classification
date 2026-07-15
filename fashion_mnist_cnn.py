"""
Module 6 Assignment: Fashion MNIST Classification
---------------------------------------------------
A 6-layer Convolutional Neural Network (CNN) built with Keras to classify
the Fashion MNIST dataset (keras.datasets.fashion_mnist).

Author: Junior ML Researcher, Microsoft AI (assignment submission)

Usage:
    python fashion_mnist_cnn.py

Requirements:
    pip install tensorflow matplotlib numpy
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers


# Human-readable names for the 10 Fashion MNIST classes.
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


class FashionMNISTClassifier:
    """
    A six-layer CNN wrapper for loading, training, evaluating, and making
    predictions on the Fashion MNIST dataset.

    The six layers of the network are:
        1. Conv2D      - learns 32 low-level feature filters
        2. MaxPooling2D - downsamples feature maps
        3. Conv2D      - learns 64 higher-level feature filters
        4. MaxPooling2D - downsamples feature maps again
        5. Flatten     - converts 2D feature maps to a 1D feature vector
        6. Dense       - final softmax output layer (10 classes)
    """

    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
        self.history = None

        # Data placeholders, populated by load_data()
        self.x_train = self.y_train = None
        self.x_test = self.y_test = None

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def load_data(self):
        """Load and preprocess the Fashion MNIST dataset."""
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

        # Normalize pixel values to [0, 1] and add the channel dimension
        # (28, 28) -> (28, 28, 1) since the images are grayscale.
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

        print(f"Training data shape: {self.x_train.shape}")
        print(f"Test data shape:     {self.x_test.shape}")
        return (x_train, y_train), (x_test, y_test)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    def _build_model(self):
        """Construct the six-layer CNN architecture."""
        model = keras.Sequential(
            [
                keras.Input(shape=self.input_shape),
                layers.Conv2D(32, kernel_size=3, activation="relu"),   # Layer 1
                layers.MaxPooling2D(pool_size=2),                      # Layer 2
                layers.Conv2D(64, kernel_size=3, activation="relu"),   # Layer 3
                layers.MaxPooling2D(pool_size=2),                      # Layer 4
                layers.Flatten(),                                      # Layer 5
                layers.Dense(self.num_classes, activation="softmax"),  # Layer 6
            ],
            name="fashion_mnist_cnn",
        )

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def summary(self):
        self.model.summary()

    # ------------------------------------------------------------------
    # Training / evaluation
    # ------------------------------------------------------------------
    def train(self, epochs=10, batch_size=128, validation_split=0.1):
        """Train the CNN on the training set."""
        if self.x_train is None:
            raise RuntimeError("Call load_data() before training.")

        self.history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
        )
        return self.history

    def evaluate(self):
        """Evaluate the CNN on the held-out test set."""
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"Test loss:     {loss:.4f}")
        print(f"Test accuracy: {accuracy:.4f}")
        return loss, accuracy

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, images):
        """Return predicted class indices for a batch of images."""
        probabilities = self.model.predict(images)
        return np.argmax(probabilities, axis=1)

    def predict_and_show(self, num_images=5):
        """
        Predict on `num_images` samples from the test set and display
        each image alongside its true and predicted label.
        """
        if self.x_test is None:
            raise RuntimeError("Call load_data() before predicting.")

        indices = np.random.choice(len(self.x_test), size=num_images, replace=False)
        sample_images = self.x_test[indices]
        true_labels = self.y_test[indices]
        predicted_labels = self.predict(sample_images)

        fig, axes = plt.subplots(1, num_images, figsize=(3 * num_images, 3))
        if num_images == 1:
            axes = [axes]

        for ax, image, true_label, pred_label in zip(
            axes, sample_images, true_labels, predicted_labels
        ):
            ax.imshow(image.squeeze(), cmap="gray")
            title_color = "green" if true_label == pred_label else "red"
            ax.set_title(
                f"True: {CLASS_NAMES[true_label]}\nPred: {CLASS_NAMES[pred_label]}",
                color=title_color,
                fontsize=10,
            )
            ax.axis("off")

        plt.tight_layout()
        plt.savefig("predictions.png", dpi=150)
        print("Saved prediction visualization to predictions.png")
        plt.show()

        for i, idx in enumerate(indices):
            print(
                f"Image {idx}: true label = {CLASS_NAMES[true_labels[i]]}, "
                f"predicted label = {CLASS_NAMES[predicted_labels[i]]}"
            )

def main():
    classifier = FashionMNISTClassifier()
    classifier.summary()

    classifier.load_data()
    classifier.train(epochs=10)
    classifier.evaluate()

    # Task requirement: make predictions for at least two images.
    classifier.predict_and_show(num_images=10)


if __name__ == "__main__":
    main()
