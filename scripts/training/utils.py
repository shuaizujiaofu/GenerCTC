import torch
from torch import Tensor, nn
from d2l.torch import d2l
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA, TruncatedSVD
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.optimize import minimize
from sklearn.svm import SVC


def evaluate_accuracy_gpu(net, data_iter, mode=None, device=None):
    """
    Compute the accuracy for a model on a dataset using a GPU.
    """
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            if mode is None:
                net_ouput = net(X)
            elif mode == "test_acc":
                net_ouput = net.test_acc(X)
            else:
                raise 'This output method does not exist'
            acc = d2l.accuracy(net_ouput, y)
            metric.add(acc, d2l.size(y))
    return metric[0] / metric[1]


def evaluate_spearman_gpu(net, data_iter, num_labels, device=None):
    """Compute the accuracy for a model on a dataset using a GPU.

    Defined in :numref:`sec_lenet`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            # labels_mask = onehot_labeling(y, num_labels)
            labels_mask = torch.eye(len(y))
            x_sim = net.test_pearson(X)
            test_spearman_sum = 0
            for i in range(labels_mask.shape[0]):
                train_spearman, _ = spearmanr(labels_mask[i].tolist(), x_sim[i].cpu().detach().numpy())
                test_spearman_sum += train_spearman
            metric.add(test_spearman_sum, labels_mask.shape[0])

    return metric[0] / metric[1]


def evaluate_pearson_gpu(net, data_iter, num_labels, device=None):
    """Compute the accuracy for a model on a dataset using a GPU.

    Defined in :numref:`sec_lenet`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            # labels_mask = onehot_labeling(y, num_labels)
            labels_mask = torch.eye(len(y))
            x_sim = net.test_pearson(X)
            test_pearson_sum = 0
            for i in range(labels_mask.shape[0]):
                train_pearson, _ = pearsonr(labels_mask[i].tolist(), x_sim[i].cpu().detach().numpy())
                test_pearson_sum += train_pearson
            metric.add(test_pearson_sum, labels_mask.shape[0])

    return metric[0] / metric[1]


def get_pc(net, train_iter):
    net.eval()
    all_feature = []
    all_label = []
    labels_pc = []
    for i, (features, labels) in enumerate(train_iter):
        with torch.no_grad():
            embeddings = net.encode(features)
            pooler_embeddings = net.pooled(embeddings, features)
            pooler_embeddings = pooler_embeddings.detach()
            all_feature.append(pooler_embeddings)
            all_label.append(labels)
    all_feature = torch.cat(all_feature, dim=0).numpy()
    all_label = torch.cat(all_label, dim=0).numpy()
    df = pd.DataFrame(all_feature)
    df.insert(loc=df.shape[1], column='label', value=all_label)
    group = df.groupby('label')
    for key, tokens in group:
        tokens.reset_index(drop=True, inplace=True)
        embeds = tokens[tokens.columns.difference(['label'])]
        pc = compute_pc(embeds, 1)
        labels_pc.append(pc)
    labels_pc = torch.cat(labels_pc, dim=0)

    return labels_pc


def compute_pc(vectors, npc):
    # removing the projection on the first principal component
    # randomized SVD version will not be affected by scale of input, see https://github.com/PrincetonML/SIF/issues/4
    # svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd = PCA(n_components=npc, svd_solver='randomized')
    svd.fit(vectors)
    pc = torch.tensor(svd.components_).float()
    return pc


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


def onehot_labeling(label_ids, num_labels):
    label_ids = label_ids.cpu()
    labels = to_categorical(label_ids, num_classes=num_labels)
    labels = torch.from_numpy(labels)
    mask = torch.mm(labels, labels.T).bool().long()
    return mask


def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    weight = (batch ** 2) / (torch.sum(batch, 0) + 1e-9)
    return (weight.t() / torch.sum(weight, 1)).t()
