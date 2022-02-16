import pytest

import tensorflow as tf

from basars_addons.schedules import CosineDecayWarmupRestarts, InitialCosineDecayRestarts


def test_initial_cosine_decay_type_mismatch():
    scheduler = InitialCosineDecayRestarts(0, 1e-3, 300, 1.0)

    epoch = tf.constant(1, dtype=tf.int64)
    learning_rate = scheduler(epoch)
    assert learning_rate.numpy() > 0.


def test_cosine_decay_warmup_type_mismatch():
    scheduler = CosineDecayWarmupRestarts(100, 1e-3, 300, 1.0)

    epoch = tf.constant(1, dtype=tf.int64)
    learning_rate = scheduler(epoch)
    assert learning_rate.numpy() > 0.


if __name__ == '__main__':
    pytest.main([__file__])
