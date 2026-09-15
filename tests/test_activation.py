import pytest
import numpy as np
from utils.activation import get_activation


@pytest.mark.parametrize("act_name", ["relu", "sigmoid", "tanh", "linear"])
def test_activation_factory_valid(act_name):
    """Test that valid activation names return callable functions."""
    func, deriv = get_activation(act_name)
    assert callable(func)
    assert callable(deriv)


def test_activation_factory_invalid():
    """Test that an unknown activation name raises a ValueError."""
    with pytest.raises(ValueError, match="Unknown activation function"):
        get_activation("invalid_act")


def test_relu():
    """Test ReLU forward and derivative behavior."""
    func, deriv = get_activation("relu")
    x = np.array([-2.0, 0.0, 3.0])

    np.testing.assert_array_equal(func(x), np.array([0.0, 0.0, 3.0]))
    np.testing.assert_array_equal(deriv(x), np.array([0.0, 0.0, 1.0]))


def test_sigmoid():
    """Test Sigmoid forward output at zero."""
    func, deriv = get_activation("sigmoid")
    x = np.array([0.0])
    np.testing.assert_allclose(func(x), np.array([0.5]))
    np.testing.assert_allclose(deriv(x), np.array([0.25]))


def test_tanh_math():
    """Test Tanh forward output at zero and limits."""
    func, deriv = get_activation("tanh")
    x = np.array([0.0])
    np.testing.assert_allclose(func(x), np.array([0.0]))
    np.testing.assert_allclose(deriv(x), np.array([1.0]))
