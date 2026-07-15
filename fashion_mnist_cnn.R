# Module 6 Assignment: Fashion MNIST Classification
# ----------------------------------------------------
# A 6-layer Convolutional Neural Network (CNN) built with Keras (R) to
# classify the Fashion MNIST dataset (keras3::dataset_fashion_mnist()).
#
# Author: Junior ML Researcher, Microsoft AI (assignment submission)
#
# Usage:
#   Rscript fashion_mnist_cnn.R
#
# Requirements (run once):
#   install.packages(c("R6", "ggplot2"))
#   install.packages("keras3")
#   keras3::install_keras()

library(R6)
library(keras3)
library(ggplot2)

# Human-readable names for the 10 Fashion MNIST classes.
CLASS_NAMES <- c(
  "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
)

#' FashionMNISTClassifier
#'
#' An R6 class that wraps loading, training, evaluating, and predicting
#' with a six-layer CNN on the Fashion MNIST dataset.
#'
#' The six layers of the network are:
#'   1. Conv2D       - learns 32 low-level feature filters
#'   2. MaxPooling2D - downsamples feature maps
#'   3. Conv2D       - learns 64 higher-level feature filters
#'   4. MaxPooling2D - downsamples feature maps again
#'   5. Flatten      - converts 2D feature maps to a 1D feature vector
#'   6. Dense        - final softmax output layer (10 classes)
FashionMNISTClassifier <- R6Class(
  "FashionMNISTClassifier",
  public = list(
    input_shape = NULL,
    num_classes = NULL,
    model = NULL,
    history = NULL,
    x_train = NULL,
    y_train = NULL,
    x_test = NULL,
    y_test = NULL,

    initialize = function(input_shape = c(28, 28, 1), num_classes = 10) {
      self$input_shape <- input_shape
      self$num_classes <- num_classes
      self$model <- private$build_model()
    },

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    load_data = function() {
      fashion_mnist <- dataset_fashion_mnist()

      x_train <- fashion_mnist$train$x
      y_train <- fashion_mnist$train$y
      x_test <- fashion_mnist$test$x
      y_test <- fashion_mnist$test$y

      # Normalize pixel values to [0, 1] and add the channel dimension:
      # (28, 28) -> (28, 28, 1) since the images are grayscale.
      x_train <- array_reshape(x_train / 255, c(dim(x_train), 1))
      x_test <- array_reshape(x_test / 255, c(dim(x_test), 1))

      self$x_train <- x_train
      self$y_train <- y_train
      self$x_test <- x_test
      self$y_test <- y_test

      cat("Training data shape:", paste(dim(self$x_train), collapse = " x "), "\n")
      cat("Test data shape:    ", paste(dim(self$x_test), collapse = " x "), "\n")

      invisible(list(
        train = list(x = x_train, y = y_train),
        test = list(x = x_test, y = y_test)
      ))
    },

    # ------------------------------------------------------------------
    # Training / evaluation
    # ------------------------------------------------------------------
    train = function(epochs = 10, batch_size = 128, validation_split = 0.1) {
      if (is.null(self$x_train)) {
        stop("Call load_data() before training.")
      }

      self$history <- self$model |> fit(
        self$x_train,
        self$y_train,
        epochs = epochs,
        batch_size = batch_size,
        validation_split = validation_split
      )
      invisible(self$history)
    },

    evaluate = function() {
      metrics <- self$model |> evaluate(self$x_test, self$y_test, verbose = 0)
      cat(sprintf("Test loss:     %.4f\n", metrics["loss"]))
      cat(sprintf("Test accuracy: %.4f\n", metrics["accuracy"]))
      invisible(metrics)
    },

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    predict_classes = function(images) {
      probabilities <- self$model |> predict(images)
      apply(probabilities, 1, which.max) - 1  # convert 1-indexed to 0-indexed
    },

    predict_and_show = function(num_images = 10) {
      if (is.null(self$x_test)) {
        stop("Call load_data() before predicting.")
      }

      n_test <- dim(self$x_test)[1]
      indices <- sample(seq_len(n_test), size = num_images)

      sample_images <- self$x_test[indices, , , , drop = FALSE]
      true_labels <- self$y_test[indices]
      predicted_labels <- self$predict_classes(sample_images)

      for (i in seq_along(indices)) {
        cat(sprintf(
          "Image %d: true label = %s, predicted label = %s\n",
          indices[i],
          CLASS_NAMES[true_labels[i] + 1],
          CLASS_NAMES[predicted_labels[i] + 1]
        ))
      }

      # Save a simple grid of the sampled images with true/predicted labels.
      png("predictions_r.png", width = 200 * num_images, height = 220)
      par(mfrow = c(1, num_images), mar = c(1, 1, 3, 1))
      for (i in seq_along(indices)) {
        img <- sample_images[i, , , 1]
        img <- t(apply(img, 2, rev))  # orient correctly for image()
        color <- ifelse(true_labels[i] == predicted_labels[i], "darkgreen", "red")
        image(
          img, col = gray.colors(256), axes = FALSE,
          main = sprintf(
            "True: %s\nPred: %s",
            CLASS_NAMES[true_labels[i] + 1],
            CLASS_NAMES[predicted_labels[i] + 1]
          ),
          col.main = color, cex.main = 0.8
        )
      }
      dev.off()
      cat("Saved prediction visualization to predictions_r.png\n")
    },

    summary = function() {
      self$model
    }
  ),
  private = list(
    build_model = function() {
      model <- keras_model_sequential(input_shape = self$input_shape) |>
        layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu") |>  # Layer 1
        layer_max_pooling_2d(pool_size = c(2, 2)) |>                                # Layer 2
        layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") |>  # Layer 3
        layer_max_pooling_2d(pool_size = c(2, 2)) |>                                # Layer 4
        layer_flatten() |>                                                          # Layer 5
        layer_dense(units = self$num_classes, activation = "softmax")               # Layer 6

      model |> compile(
        optimizer = "adam",
        loss = "sparse_categorical_crossentropy",
        metrics = "accuracy"
      )
      model
    }
  )
)

main <- function() {
  classifier <- FashionMNISTClassifier$new()
  print(classifier$summary())

  classifier$load_data()
  classifier$train(epochs = 10)
  classifier$evaluate()

  # Task requirement: make predictions for at least two images.
  classifier$predict_and_show(num_images = 10)
}

if (sys.nframe() == 0) {
  main()
}
