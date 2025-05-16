import numpy as np
from decisionTree.tree import build_tree
from decisionTree.predictions import predict
from scipy.stats import mode

class Random_Forest_Classifier:
    def __init__(self, n_trees = 10, max_depth = 10, min_samples_split = 2, feature_subset_size = None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_subset_size = feature_subset_size
        self.trees = []

    def allocate_trees(self, X, y):
        for _ in range(self.n_trees):
            X_sample, y_sample = bootstrap_sample(X, y)
            small_tree = build_tree(X_sample, y_sample, 0, self.max_depth, self.min_samples_split, self.feature_subset_size)
            self.trees.append(small_tree)

    def forest_predict(self, X):
        all_preds = [predict(X, tree) for tree in self.trees]
        return majority_vote(all_preds)


def bootstrap_sample(X, y):
    indices = np.random.choice(len(X), size=len(X), replace=True)
    return X[indices], y[indices]

def majority_vote(preds):
    preds = np.array(preds)
    preds = preds.T # transpose this matrix 
    final_preds, _ = mode(preds, axis=1)
    return final_preds.flatten()