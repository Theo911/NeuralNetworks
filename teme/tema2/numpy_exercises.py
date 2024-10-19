from functools import wraps
import time
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from typing import Tuple

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper



# ------------------------------------------------------
@timeit
def tp_fp_fn_tn_sklearn(gt: np.ndarray, pred: np.ndarray) -> Tuple[int, ...]:
    tn, fp, fn, tp = confusion_matrix(gt, pred).ravel()
    return tp, fp, fn, tn


# --------------------- Exercise 1 ---------------------
@timeit
def tp_fp_fn_tn_numpy(gt: np.ndarray, pred: np.ndarray) -> Tuple[int, ...]:
    """Exercise 1
        Implement a method to retrieve the confusion matrix values using numpy operations.
        Aim to make your method faster than the sklearn implementation.
    """

    tp = int(np.sum(np.logical_and(gt == 1, pred == 1)))
    fp = int(np.sum(np.logical_and(gt == 0, pred == 1)))
    fn = int(np.sum(np.logical_and(gt == 1, pred == 0)))
    tn = int(np.sum(np.logical_and(gt == 0, pred == 0)))

    return tp, fp, fn, tn


# ------------------------------------------------------
@timeit
def accuracy_sklearn(gt: np.ndarray, pred: np.ndarray) -> float:
    return accuracy_score(gt, pred)


# --------------------- Exercise 2 ---------------------
@timeit
def accuracy_numpy(gt: np.ndarray, pred: np.ndarray) -> float:
    """Exercise 2
        Implement a method to retrieve the calculate the accuracy using numpy operations.
        Accuracy is the proportion of true results (both true positives and true negatives) among the total number of cases examined.
    """
    tp, fp, fn, tn = tp_fp_fn_tn_numpy(gt, pred)
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    return accuracy


# ------------------------------------------------------
@timeit
def f1_score_sklearn(gt: np.ndarray, pred: np.ndarray) -> float:
    return f1_score(gt, pred)


# --------------------- Exercise 3 ---------------------
# Implement a method to calculate the F1-Score using numpy operations.
# Be careful at corner cases (divide by 0).

def precision_numpy(gt: np.ndarray, pred: np.ndarray) -> float:
    """Precision is a measure of how accurate a model’s positive predictions are."""
    tp, fp, fn, tn = tp_fp_fn_tn_numpy(gt, pred)
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    return precision

def recall_numpy(gt: np.ndarray, pred: np.ndarray) -> float:
    """ Recall is a measure of how many of the actual positives our model capture through labeling it as Positive."""
    tp, fp, fn, tn = tp_fp_fn_tn_numpy(gt, pred)
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    return recall

def f1_score_numpy(gt: np.ndarray, pred: np.ndarray) -> float:
    """The F1 score is the harmonic mean of precision and recall."""
    precision = precision_numpy(gt, pred)
    recall = recall_numpy(gt, pred)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    return f1


if __name__ == '__main__':
    predicted = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    actual = np.array([1, 1, 1, 1, 0, 0, 1, 0, 0, 0])

    big_size = 500000
    big_actual = np.repeat(actual, big_size)
    big_predicted = np.repeat(predicted, big_size)

    # Exercise 1
    # rez_1 = tp_fp_fn_tn_sklearn(big_actual, big_predicted)
    # rez_2 = tp_fp_fn_tn_numpy(big_actual, big_predicted)
    # print(rez_1)
    # print(rez_2)

    # Exercise 2
    # rez_1 = accuracy_sklearn(big_actual, big_predicted)
    # rez_2 = accuracy_numpy(big_actual, big_predicted)
    # assert np.isclose(rez_1, rez_2)

    # Exercise 3
    # rez_1 = f1_score_sklearn(big_actual, big_predicted)
    # rez_2 = f1_score_numpy(big_actual, big_predicted)
    # assert np.isclose(rez_1, rez_2)