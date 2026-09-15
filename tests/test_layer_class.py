import numpy as np
from layer_class import Layer


def test_layer_initialization():
    """Test that a layer initializes weights, biases, and shapes properly."""
    layer = Layer(n_in=4, n_out=3, activation="relu", initializer="he")

    assert layer.weights.shape == (3, 4)
    assert layer.biases.shape == (1, 3)
    assert callable(layer.activation)
    assert callable(layer.activation_derived)


def test_layer_forward_pass():
    """Test the forward pass output shape and data flow."""
    layer = Layer(n_in=3, n_out=2, activation="linear", initializer="zero")
    layer.weights = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    layer.biases = np.array([[1.0, -1.0]])

    X = np.array([[1.0, 1.0, 1.0], [2.0, 0.0, 1.0]])

    output = layer.forward(X)
    # Expected Z = XW + B:
    # Sample 1: [1+3+5+1, 2+4+6-1] = [10, 11]
    # Sample 2: [2+0+5+1, 4+0+6-1] = [8, 9]
    expected = np.array([[10.0, 11.0], [8.0, 9.0]])

    np.testing.assert_allclose(output, expected)


def test_layer_backward_pass():
    """Test that the backward pass computes correct shapes and updates weights."""
    layer = Layer(n_in=2, n_out=2, activation="linear", initializer="zero")
    layer.weights = np.array([[1.0, 2.0], [3.0, 4.0]])
    dummy_output_grad = np.array([[0.5, -0.5]])
    X = np.array([[1.0, 2.0]])
    _ = layer.forward(X)
    # Run backward pass (with learning_rate = 0.0 to inspect raw gradients before update)
    input_grad = layer.backward(dummy_output_grad, learning_rate=0.0)

    # --- Manual Calculations for Verification ---
    # Linear activation derivative is 1 everywhere, so dz = dummy_output_grad = [[0.5, -0.5]]

    # Expected d_weights = dz.T @ X -> (2x1) @ (1x2) = (2x2)
    # dz.T = [[0.5], [-0.5]]
    # X = [[1.0, 2.0]]
    # Expected d_weights = [[0.5*1.0, 0.5*2.0], [-0.5*1.0, -0.5*2.0]] = [[0.5, 1.0], [-0.5, -1.0]]
    expected_d_weights = np.array([[0.5, 1.0], [-0.5, -1.0]])

    # Expected d_biases = sum of dz across batch -> [[0.5, (-0.5)]]
    expected_d_biases = np.array([[0.5, -0.5]])

    # Expected input_gradient = dz @ weights -> (1x2) @ (2x2) = (1x2)
    # [[0.5, -0.5]] @ [[1.0, 2.0], [3.0, 4.0]] = [[0.5*1.0 + (-0.5)*3.0, 0.5*2.0 + (-0.5)*4.0]]
    # = [[0.5 - 1.5, 1.0 - 2.0]] = [[-1.0, -1.0]]
    expected_input_grad = np.array([[-1.0, -1.0]])

    # --- Assertions ---
    np.testing.assert_allclose(layer.d_weights, expected_d_weights)
    np.testing.assert_allclose(layer.d_biases, expected_d_biases)
    np.testing.assert_allclose(input_grad, expected_input_grad)
