import torch
from torch import nn, Tensor
import numpy as np


def mean_pooling(model_output, attention_mask):
    """
    Mean Pooling - Take attention mask into account for correct averaging
    First element of model_output contains all token embeddings
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
    return torch.sum(model_output * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def max_pooling(model_output, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
    model_output[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
    max_over_time = torch.max(model_output, 1)[0]
    return max_over_time


def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

     E.g. for use with categorical_crossentropy.

     # Arguments
         y: class vector to be converted into a matrix
             (integers from 0 to num_classes).
         num_classes: total number of classes.
         dtype: The data type expected by the input, as a string
             (`float32`, `float64`, `int32`...)

     # Returns
         A binary matrix representation of the input. The classes axis
         is placed last.

     # Example

     ```python
     # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
     > labels
     array([0, 2, 1, 2, 0])
     # `to_categorical` converts this into a matrix with as many
     # columns as there are classes. The number of rows
     # stays the same.
     > to_categorical(labels)
     array([[ 1.,  0.,  0.],
            [ 0.,  0.,  1.],
            [ 0.,  1.,  0.],
            [ 0.,  0.,  1.],
            [ 1.,  0.,  0.]], dtype=float32)
     ```
     """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
