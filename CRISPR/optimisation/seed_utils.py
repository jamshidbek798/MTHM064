"""
Set global seeds for reproducibility across Python, NumPy, and TensorFlow.

This ensures deterministic behaviour where possible, including GPU operations.
"""

import os
import random

import numpy as np
import tensorflow as tf

def set_seed(seed=42):
    """
    Set random seed for reproducibility.

    Parameters
    ----------
    seed : int
        Seed value for all random number generators.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)  # Python hash seed
    os.environ["TF_DETERMINISTIC_OPS"] = "1"  # TensorFlow deterministic ops
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # CuDNN deterministic mode
    random.seed(seed)  # Python RNG
    np.random.seed(seed)  # NumPy RNG
    tf.random.set_seed(seed)  # TensorFlow RNG

    # Enable deterministic ops in TF 2.x (if available)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as e:
        print("Determinism already enabled or unsupported:", e)