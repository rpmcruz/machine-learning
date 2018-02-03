import numpy as np
import sys
sys.setrecursionlimit(5000000)


def f(index, current_label, labels, num_labels, dp_matrix, class_weight):
    if index >= len(labels) or current_label >= num_labels:
        return 0

    if dp_matrix[index][current_label] != -1:
        return dp_matrix[index][current_label]

    error = class_weight[labels[index]][current_label]

    if current_label + 1 == num_labels:
        dp_matrix[index][current_label] = \
            error + \
            f(index + 1, current_label, labels, num_labels, dp_matrix, class_weight)
    else:
        dp_matrix[index][current_label] = \
            min(error +
                f(index + 1, current_label, labels, num_labels, dp_matrix, class_weight),
                f(index, current_label + 1, labels, num_labels, dp_matrix, class_weight))
    return dp_matrix[index][current_label]


def _decide_thresholds(scores, labels, num_labels, class_weight):
    def traverse_matrix(dp_matrix, class_weight):
        nscores, nlabels = dp_matrix.shape
        index, current_label = 0, 0
        ret = []
        while index+1 < nscores and current_label+1 < num_labels:
            current = dp_matrix[index][current_label]
            keep = dp_matrix[index + 1][current_label]
            error = class_weight[labels[index]][current_label]
            if abs((current - error) - keep) < 1e-5:
                index += 1
            else:
                ret.append(index)
                current_label += 1
        return ret

    dp_matrix = -np.ones((len(labels), num_labels), dtype=np.float32)
    f(0, 0, labels, num_labels, dp_matrix, class_weight)
    path = traverse_matrix(dp_matrix, class_weight)

    #return scores[path]  # old behavior: return midpoints
    return np.asarray([(scores[p]+scores[max(p-1, 0)])/2 for p in path])


def decide_thresholds(scores, y, K, strategy):
    if strategy == 'uniform':
        w = 1-np.eye(K)
    elif strategy == 'inverse':
        w = np.repeat(len(y) / (K*(np.bincount(y)+1)), K).reshape((K, K)) * (1-np.eye(K))
    elif strategy == 'absolute':
        w = [[np.abs(i-j) for i in range(K)] for j in range(K)]
    elif strategy == 'absolute_inverse':
        w1 = np.repeat(len(y) / (K*(np.bincount(y)+1)), K).reshape((K, K)) * (1-np.eye(K))
        w2 = [[np.abs(i-j) for i in range(K)] for j in range(K)]
        w = w1*w2
    else:
        raise 'No such threshold strategy: %s' % strategy

    return _decide_thresholds(scores, y, K, w)
