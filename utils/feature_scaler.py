import numpy as np
from numpy.typing import NDArray


class Feature_Scaler:
    def __init__(self):
        self.mean: NDArray[np.float64] | None = None
        self.std: NDArray[np.float64] | None = None

    def set(self, mean: NDArray[np.float64], std: NDArray[np.float64]) -> None:
        self.mean = mean
        self.std = std

    def fit(self, X: NDArray[np.float64]) -> None:
        self.mean = np.nanmean(X, axis=0)
        self.std = np.nanstd(X, axis=0)

        self.std[self.std == 0.0] = 1e-15

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler must be fitted before calling transform")

        nan_mask = np.isnan(X)
        normalized = (X - self.mean) / self.std
        normalized[nan_mask] = 0.0
        return normalized

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        self.fit(X)
        return self.transform(X)
