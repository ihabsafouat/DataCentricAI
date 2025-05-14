# src/data_validation.py

import numpy as np
from tensorflow.keras.datasets import fashion_mnist

def validate_data():
    (x_train, y_train), _ = fashion_mnist.load_data()
    assert x_train.shape[1:] == (28, 28), "Unexpected image size!"
    assert x_train.dtype == np.uint8, "Unexpected data type!"
    assert len(set(y_train)) == 10, "Expected 10 classes!"

    print("✅ Data validation passed.")

if __name__ == "__main__":
    validate_data()
