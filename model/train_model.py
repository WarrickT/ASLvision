import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from decisionTree.tree import build_tree
from decisionTree.predictions import predict
from model.randomForest import Random_Forest_Classifier
import pickle

def run():

    df = pd.read_csv('./data/asl_landmarks_regular.csv')
    x = df.drop('label', axis=1)
    x = x.apply(pd.to_numeric, errors='coerce')
    x = x.fillna(0)

    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(x.values, y.values, test_size=0.2, random_state=42)

    max_depth = 15
    min_samples_split = 10

    n_trees = 15
    feature_subset_size = int(np.sqrt(X_train.shape[1]))
    rf = Random_Forest_Classifier(n_trees=n_trees, max_depth=max_depth, min_samples_split=min_samples_split, feature_subset_size=feature_subset_size)
    rf.allocate_trees(X_train, y_train)
    preds = rf.forest_predict(X_test)

    accuracy = np.mean(np.array(preds) == np.array(y_test))
    print(f"Accuracy: {accuracy:.4f}")
    print("Sample predictions:", preds[:10])
    print("Actual labels:     ", y_test[:10])

    with open("model/randomforest.pkl", "wb") as f:
        pickle.dump(rf, f)
    #train_test_split
