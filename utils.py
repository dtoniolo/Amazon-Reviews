import numpy as np
import pandas as pd


def class_counts(labels):
    unique, counts = np.unique(labels, return_counts=True)
    ind = np.argsort(unique)
    unique = unique[ind]
    counts = counts[ind]
    return unique, counts


def _print_class_counts(unique, counts):
    df = pd.DataFrame({'Class': unique, 'Counts': counts,
                       'Rel Counts (%)': 100*counts/counts.sum()})
    df = df.T
    print(df)


def print_class_counts(labels):
    print('Class frequencies:')
    _print_class_counts(*class_counts(labels))


def get_scores(counts, classes, scores):
    weights = 1/counts
    weights = weights / weights.sum()
    scores = scores.reshape(scores.size, 1)
    classes = classes[np.newaxis, :]
    weights = weights[np.where(scores == classes)[1]]
    return weights
