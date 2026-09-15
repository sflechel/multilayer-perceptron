import pytest
import numpy as np
from layer import DenseLayer


def test_layer_initialization():
    """Test that a layer initializes weights, biases, and shapes properly."""
    layer = DenseLayer(n_in=4, n_out=3, activation="relu", initializer="he")

    assert layer.weights.shape == (4, 3)
    assert layer.biases.shape == (1, 3)
    assert callable(layer.activation_func)
    assert callable(layer.activation_deriv)


def test_layer_forward_pass():
    """Test the forward pass output shape and data flow."""
    layer = DenseLayer(n_in=3, n_out=2, activation="linear", initializer="zero")
    # Set custom weights and biases for deterministic math checking
    layer.weights = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    layer.biases = np.array([[1.0, -1.0]])

    # Input batch of 2 samples, 3 features each
    X = np.array([[1.0, 1.0, 1.0], [2.0, 0.0, 1.0]])

    output = layer.forward(X)
    # Expected Z = XW + B:
    # Sample 1: [1+3+5+1, 2+4+6-1] = [10, 11]
    # Sample 2: [2+0+5+1, 4+0+6-1] = [8, 9]
    expected = np.array([[10.0, 11.0], [8.0, 9.0]])

    np.testing.assert_allclose(output, expected)


def test_layer_backward_pass():
    """Test that the backward pass computes correct shapes and updates weights."""
    layer = DenseLayer(n_in=2, n_out=2, activation="linear", initializer="zero")
    X = np.array([[1.0, 2.0]])

    _ = layer.forward(X)
    dummy_output_grad = np.array([[0.5, -0.5]])

    input_grad = layer.backward(dummy_output_grad, learning_rate=0.1)

    # Check that gradients and shapes match expectation
    assert layer.d_weights.shape == (2, 2)
    assert layer.d_biases.shape == (1, 2)
    assert input_grad.shape == (1, 2)
