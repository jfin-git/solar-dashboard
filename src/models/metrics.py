
import numpy as np

def daylight_mask(y, threshold=5.0):
    return y > threshold

def mae(y_true, y_pred, mask=None):
    if mask is None: mask = np.ones_like(y_true, dtype=bool)
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))

def rmse(y_true, y_pred, mask=None):
    if mask is None: mask = np.ones_like(y_true, dtype=bool)
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2)))

def mape(y_true, y_pred, mask=None):
    if mask is None: mask = np.ones_like(y_true, dtype=bool)
    denom = np.clip(np.abs(y_true[mask]), 1e-3, None)
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / denom)) * 100.0)
