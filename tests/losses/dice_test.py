import pytest
import numpy as np

from basars_addons.losses import Dice


def sample_true_pred():
    y_true = np.array([[1, 1, 1], [0, 1, 0]], dtype=np.float32)
    y_true = np.reshape(y_true, (2, 1, -1, 1))
    y_pred = np.array([[0, 0, 0], [1, 0, 1]], dtype=np.float32)
    y_pred = np.reshape(y_pred, (2, 1, -1, 1))

    return y_true, y_pred


def test_dice():
    dice = Dice(name='dice')

    y_true, y_pred = sample_true_pred()

    assert dice(y_true, y_pred).numpy() >= 0.999
    assert dice(y_true, y_true).numpy() == 0.



if __name__ == '__main__':
    pytest.main([__file__])
