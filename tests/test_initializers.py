import pytest
import numpy as np
from utils.initializers import get_initializer, xavier_init, he_init, zero_init


@pytest.mark.parametrize("init_name", ["xavier", "he", "zero"])
def test_initializer_factory_valid(init_name):
    """Test that valid initializer names return callable functions."""
    func = get_initializer(init_name)
    assert callable(func)


def test_initializer_factory_invalid():
    """Test that an unknown initializer name raises a ValueError."""
    with pytest.raises(ValueError, match="Unknown initializer"):
        get_initializer("bad_initializer")


def test_xavier_shape_and_scale():
    """Test Xavier initialization shape and bounds."""
    n_in, n_out = 10, 5
    weights = xavier_init(n_in, n_out)

    assert weights.shape == (n_out, n_in)
    limit = np.sqrt(6.0 / (n_in + n_out))
    assert np.all(weights >= -limit)
    assert np.all(weights <= limit)


def test_he_shape():
    """Test He initialization shape."""
    n_in, n_out = 20, 10
    weights = he_init(n_out, n_in)
    assert weights.shape == (n_in, n_out)


def test_zero_shape_and_values():
    """Test zero initialization creates a matrix of exact zeros."""
    n_in, n_out = 5, 5
    weights = zero_init(n_in, n_out)
    assert weights.shape == (n_out, n_in)
    np.testing.assert_array_equal(weights, np.zeros((n_in, n_out)))
