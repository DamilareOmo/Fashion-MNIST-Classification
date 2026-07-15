# Fashion MNIST Classification

A 6-layer Convolutional Neural Network (CNN) built with Keras, implemented in
both **Python** and **R**, to classify the [Fashion MNIST](https://keras.io/api/datasets/fashion_mnist/)
dataset. Includes predictions on sample test images.

## Repository Structure

```
fashion_mnist_assignment/
├── README.md
├── python/
│   ├── fashion_mnist_cnn.py       # Python implementation (Keras class-based)
│   └── fashion_mnist_cnn.ipynb    # Same code as a Jupyter notebook
└── r/
    ├── fashion_mnist_cnn.R        # R implementation (keras3, R6 class-based)
```

## Dataset

The [Fashion MNIST](https://keras.io/api/datasets/fashion_mnist/) dataset
consists of 60,000 training images and 10,000 test images (28x28 grayscale)
across 10 clothing categories: T-shirt/top, Trouser, Pullover, Dress, Coat,
Sandal, Shirt, Sneaker, Bag, and Ankle boot. It is loaded directly through
Keras's built-in dataset loader, so no manual download is required.

## Model Architecture

Both implementations use the identical six-layer CNN architecture:

| # | Layer | Purpose |
|---|-------|---------|
| 1 | `Conv2D` (32 filters, 3x3, ReLU) | Learns low-level features (edges, textures) |
| 2 | `MaxPooling2D` (2x2) | Downsamples feature maps, reduces overfitting |
| 3 | `Conv2D` (64 filters, 3x3, ReLU) | Learns higher-level features |
| 4 | `MaxPooling2D` (2x2) | Further downsampling |
| 5 | `Flatten` | Converts 2D feature maps into a 1D vector |
| 6 | `Dense` (10 units, softmax) | Final classification output |

Optimizer: `adam` · Loss: `sparse_categorical_crossentropy` · Metric: `accuracy`

## Python Instructions

### Requirements
```bash
pip install tensorflow matplotlib numpy
```

### Run
```bash
cd python
python fashion_mnist_cnn.py
```

### What it does
1. Loads and normalizes the Fashion MNIST dataset via `FashionMNISTClassifier.load_data()`.
2. Builds and trains the six-layer CNN for 10 epochs (`.train()`).
3. Evaluates accuracy on the 10,000-image test set (`.evaluate()`).
4. Predicts labels for 5 random test images and saves a labeled image
   grid to `predictions.png`, printing each true vs. predicted label to
   the console (`.predict_and_show()`), satisfying the "at least two
   images" prediction requirement.

### Jupyter notebook version
`fashion_mnist_cnn.ipynb` contains the identical class and workflow broken
into cells (imports → class definition → instantiate → load data → train →
evaluate → predict). Launch with:
```bash
jupyter notebook fashion_mnist_cnn.ipynb
```

## R Instructions

### Requirements
```r
install.packages(c("R6", "ggplot2"))
install.packages("keras3")
keras3::install_keras()   # installs the TensorFlow backend, run once
```

### Run
```bash
cd r
Rscript fashion_mnist_cnn.R
```

### What it does
Mirrors the Python workflow using an `R6` class (`FashionMNISTClassifier`):
loads and normalizes the data, builds/trains the same six-layer CNN,
evaluates on the test set, and predicts + visualizes 10 sample images,
saving the result to `predictions_r.png`.

## Notes

- Both scripts are class-based (Python `class`, R `R6::R6Class`) as required
  by the assignment, keeping data loading, model building, training,
  evaluation, and prediction as clearly separated methods.
- Training for 10 epochs on Fashion MNIST typically reaches ~90-92% test
  accuracy on a CPU in a few minutes; a GPU will train much faster.
- Random seeds are not fixed, so exact accuracy and sampled prediction
  images will vary slightly between runs.

## Author

Sodiq Omoniyi | Module 6 Assignment: Fashion MNIST Classification
