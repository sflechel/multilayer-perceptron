import numpy as np
from numpy.typing import NDArray


def xavier_init(n_in: int, n_out: int, seed: int) -> NDArray[np.float64]:
    limit = np.sqrt(6.0 / (n_in + n_out))
    rng = np.random.default_rng(seed)
    return rng.uniform(-limit, limit, size=(n_out, n_in))


def he_init(n_in: int, n_out: int, seed: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    stddev = np.sqrt(2.0 / n_in)
    return rng.normal(0.0, stddev, size=(n_out, n_in))


def zero_init(n_in: int, n_out: int, seed: int) -> NDArray[np.float64]:
    return np.zeros((n_out, n_in))


def get_initializer(name: str):
    """Factory helper to fetch initializers by string name via CLI."""
    initializers = {"xavier": xavier_init, "he": he_init, "zero": zero_init}
    if name.lower() not in initializers:
        raise ValueError(
            f"Unknown initializer: {name}. Choose from {list(initializers.keys())}"
        )
    return initializers[name.lower()]
