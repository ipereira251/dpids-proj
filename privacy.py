import numpy as np

def add_gaussian_noise(X, sigma=0.1):
    feature_std = np.std(X, axis=0)
    feature_std = feature_std / (np.max(feature_std) + 1e-9)

    noise = np.random.normal(
        0,
        sigma * feature_std,
        X.shape
    )

    return X + noise