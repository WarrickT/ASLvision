from .tree import DecisionNode, build_tree
import pandas as pd
import numpy as np
from typing import List

def predict(X, root_node):
    #Utilize the decision tree from before. 
    predictions = []

    for row in range(X.shape[0]):
        x = X[row]
        label = predict_single(x, root_node)
        predictions.append(label)

    return predictions
    

def predict_single(X, node):
    while node.value is None: 
        feature_value = X[node.feature_index]
        if feature_value < node.threshold: 
            node = node.left
        else:
            node = node.right

    return node.value
    

