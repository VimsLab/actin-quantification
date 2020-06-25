from sklearn import metrics
import numpy as np
import pdb

def f1(gt, out):
    return metrics.f1_score(gt, out, average='micro')

def accuracy(gt, predict):
    return float(np.sum(gt == predict)) / len(predict)

def confusion(y_true, y_pred):
    return metrics.confusion_matrix(y_true, y_pred)

def topk(k, gt, out):

    out = out[:, :k]
    judge = (gt.reshape((out.shape[0], 1)) == out).astype(np.int32)

    return float(judge.sum()) / len(gt)

def mean_square_error(gt, out):
    return metrics.mean_squared_error(gt, out)

def mean_absolute_error(gt, out):
    return metrics.mean_absolute_error(gt, out)
