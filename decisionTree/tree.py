import numpy as np

class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


def build_tree(X, y, depth=0, max_depth=0, min_samples_split=0, feature_subset_size=None):
    # Check for purity
    if np.unique(y).size == 1:
        print(f"[Found leaf!] Depth: {depth}, Samples: {len(y)}, Label: {y[0]}")
        return DecisionNode(value=y[0])
    
    # Stopping conditions
    if len(y) < min_samples_split or depth >= max_depth:
        majority = most_common_label(y)
        print(f"[Stop] Depth: {depth}, Samples: {len(y)}, Majority Label: {majority}")
        return DecisionNode(value=majority)

    num_features = X.shape[1]
    if(feature_subset_size is None):
        feature_subset_size = num_features

    random_features = np.random.choice(num_features, feature_subset_size, replace=False)
    best_gini = float("inf")
    best_feature = None
    best_threshold = None
    best_splits = None

    for feature in random_features:
        #print(f"Testing feature {feature} at depth {depth}...")  
        values = np.unique(X[:, feature])

        thresholds = return_thresholds(values)
        #print(f"  Feature {feature} has {len(thresholds)} thresholds")


        for i, threshold in enumerate(thresholds):
            #if(i % 500 == 0):
            #   print(f"    Threshold {i}/{len(thresholds)} → {threshold:.4f}")
            left_mask = X[:, feature] < threshold
            right_mask = ~left_mask

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            y_left = y[left_mask]
            y_right = y[right_mask]

            gini = compute_Gini_split(y_left, y_right)
            if gini < best_gini:
                best_gini = gini
                best_feature = feature
                best_threshold = threshold
                best_splits = (
                    X[left_mask], y_left,
                    X[right_mask], y_right
                )

    if best_splits is None:
        return DecisionNode(value=most_common_label(y))

    X_left, y_left, X_right, y_right = best_splits
    left_subtree = build_tree(X_left, y_left, depth + 1, max_depth, min_samples_split)
    right_subtree = build_tree(X_right, y_right, depth + 1, max_depth, min_samples_split)

    print(f"[Split] Depth: {depth}, Feature: {best_feature}, Threshold: {best_threshold:.4f}, Gini: {best_gini:.4f}")

    return DecisionNode(
        feature_index=best_feature,
        threshold=best_threshold,
        left=left_subtree,
        right=right_subtree
    )


def return_thresholds(values):
    # Use quantiles for now instead of every possible midpoint
    quantiles = np.quantile(values, np.linspace(0.05, 0.95, num=15))
    return list(np.unique(quantiles))


def most_common_label(y):
    values, counts = np.unique(y, return_counts=True)
    return values[np.argmax(counts)]


def compute_Gini_split(y_left, y_right):
    n = len(y_left) + len(y_right)
    gini_left = compute_Gini_impurity(y_left)
    gini_right = compute_Gini_impurity(y_right)
    return (len(y_left) / n) * gini_left + (len(y_right) / n) * gini_right


def compute_Gini_impurity(y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs ** 2)
