import gzip
import json
import numpy as np
import pandas as pd


def read_file(filename):
    """Reads file.
    It is capable to decompressing the file if it is gzipped.

    Parameters
    ----------
    filename : str
        Path of file.

    Returns
    -------
    reviews: list
        List of reviews
    """
    filename = '../Clothing_Shoes_and_Jewelry_5.json'
    if filename.endswith('gz'):
        file_reader = gzip.open(filename, 'r')
        reviews = [json.loads(review.decode('utf-8')) for review in file_reader]
    else:
        with open(filename, 'r') as f:
            reviews = [json.loads(review) for review in f.readlines()]
    return reviews


def class_counts(labels):
    """Return sorted class labels and count

    Parameters
    ----------
    labels : np 1D array
        Array of class labels. The class labels can be any data type (numbers
        of strings)

    Returns
    -------
    unique: : np 1D array
        Array of sorted class labels
    counts : np 1D array
        Counts of the unique occurences of each class.

    """
    unique, counts = np.unique(labels, return_counts=True)
    ind = np.argsort(unique)
    unique = unique[ind]
    counts = counts[ind]
    return unique, counts


def _print_class_counts(unique, counts):
    """Pretty prints the class counts. Its inputs are the outputs of
    `class_counts`. Backend function for `print_class_counts`

    Parameters
    ----------
    unique: : np 1D array
        Array of sorted class labels
    counts : np 1D array
        Counts of the unique occurences of each class.

    Returns
    -------
    None

    """
    df = pd.DataFrame({'Class': unique, 'Counts': counts,
                       'Rel Counts (%)': 100*counts/counts.sum()})
    df = df.T
    print(df)


def print_class_counts(labels):
    """Computes and prints class

    Parameters
    ----------
    labels : type
        Description of parameter `labels`.

    Returns
    -------
    type
        Description of returned object.

    """
    print('Class frequencies:')
    _print_class_counts(*class_counts(labels))


def get_label_scores(labels, classes=None, counts=None):
    """Returns sample weights used to get a balenced classifier fit out of a
    unbalanced dataset. If classes and counts are `None` they will be computed
    automatically.

    Parameters
    ----------
    labels : np 1D array
        Array of class labels. The class labels can be any data type (numbers
        of strings)
    classes: : np 1D array
        Array of sorted class labels. 1st output of the `class_counts`
        function.
    counts : np 1D array
        Counts of the unique occurences of each class. 2nd output of the
        `class_counts` function.

    Returns
    -------
    weights : np 1D array
        Weights for each sample

    """
    if (classes is None and counts is not None) or \
       (classes is not None and counts is None):
        raise AttributeError('classes and counts should be either both' +
                             ' None, or neither should be.')
    if classes is None and counts is None:
        classes, counts = class_counts(labels)
    weights = 1/counts
    weights = weights / weights.sum()
    labels = labels.reshape(labels.size, 1)
    classes = classes[np.newaxis, :]
    weights = weights[np.where(labels == classes)[1]]
    return weights


def get_class_scores(classes=None, counts=None):
    """Returns class weights used to get a balenced classifier fit out of a
    unbalanced dataset. If classes and counts are `None` they will be computed
    automatically.

    Parameters
    ----------
    classes: : np 1D array
        Array of sorted class labels. 1st output of the `class_counts`
        function.
    counts : np 1D array
        Counts of the unique occurences of each class. 2nd output of the
        `class_counts` function.

    Returns
    -------
    weights : np 1D array
        Weights for each sample

    """
    if (classes is None and counts is not None) or \
       (classes is not None and counts is None):
        raise AttributeError('classes and counts should be either both' +
                             ' None, or neither should be.')
    weights = 1/counts
    weights = weights / weights.sum()
    return weights
