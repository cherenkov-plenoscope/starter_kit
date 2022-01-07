from plenoirf.utils import apply_confusion_matrix
import numpy as np
import pytest


def test_apply_unity_confusion_matrix():
    x = np.array([1, 2, 3,])
    cm = np.eye(3)
    y = apply_confusion_matrix(x, cm)
    assert y[0] == x[0]
    assert y[1] == x[1]
    assert y[2] == x[2]


def test_apply_confusion_matrix():
    x = np.array([1, 2, 3,])
    cm = np.array([[0.9, 0.1, 0.0], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8],])
    y = apply_confusion_matrix(x, cm)
    assert y[0] == x[0] * 0.9 + x[1] * 0.2 + x[2] * 0.1
    assert y[1] == x[0] * 0.1 + x[1] * 0.7 + x[2] * 0.1
    assert y[2] == x[0] * 0.0 + x[1] * 0.1 + x[2] * 0.8


def test_confusion_matrix_bad_dimensions():
    with pytest.raises(AssertionError) as err:
        x = np.array([1, 2])
        cm = np.array([[1,]])
        y = apply_confusion_matrix(x, cm)


def test_confusion_matrix_not_normalized():
    x = np.array([1, 2, 3,])
    cm = np.array([[0.9, 0.1, 0.0], [0.1, 0.7, 0.1], [0.0, 0.2, 0.9],])
    with pytest.raises(AssertionError) as err:
        y = apply_confusion_matrix(x, cm)
    y = apply_confusion_matrix(x, cm.T)
