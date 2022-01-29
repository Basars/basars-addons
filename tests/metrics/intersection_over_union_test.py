import cv2
import pytest
import numpy as np

from basars_addons.metrics import ThresholdBinaryIoU


IMG_SIZE = 128
LEVEL_PADS = [(0.25, 0),
              (0.55, 12),
              (0.75, 24),
              (1.0, 36)]


def apply_leveled_circles(image, radius, color_level):
    return cv2.circle(image,
                      center=(IMG_SIZE // 2, IMG_SIZE // 2),
                      radius=int(radius),
                      color=int(255 * color_level),
                      thickness=-1)


def test_intersection_over_union():
    y_pred = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)
    for level, pad in LEVEL_PADS:
        y_pred = apply_leveled_circles(y_pred, IMG_SIZE / 2 - pad, level)
    y_pred = y_pred / 255.

    for index, threshold in enumerate([0.2, 0.5, 0.7, 1.0]):
        pair = LEVEL_PADS[index]

        y_true = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)
        y_true = apply_leveled_circles(y_true, IMG_SIZE / 2 - pair[1], 1.0)
        y_true = y_true / 255.

        binary_iou = ThresholdBinaryIoU(num_classes=2, threshold=threshold, name='binary_iou')
        binary_iou.update_state(y_true, y_pred)

        assert binary_iou.result().numpy() == 1.0, 'threshold={}, level={}, pad={}'.format(threshold, *pair)


if __name__ == '__main__':
    pytest.main([__file__])
