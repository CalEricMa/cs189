"""
To prepare the starter code, copy this file over to decision_tree_starter.py
and go through and handle all the inline TODOs.
"""
from collections import Counter

import numpy as np
from numpy import genfromtxt
import scipy.io
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
import pandas as pd
from pydot import graph_from_dot_data
import io

import random
random.seed(246810)
np.random.seed(246810)

eps = 1e-5  # a small number


class DecisionTree:

    def __init__(self, max_depth=3, feature_labels=None):
        self.max_depth = max_depth
        self.features = feature_labels
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes

    @staticmethod
    def entropy(y):
        # TODO
        if len(y) == 0:
            return 0
        values, counts = np.unique(y, return_counts=True)
        p = counts/counts.sum()

        return -np.sum(p * np.log2(p + eps)) #add eps to avoid log(0)

    @staticmethod
    def information_gain(X, y, thresh):
        # TODO
        cur_entropy = DecisionTree.entropy(y)
        left = X < thresh
        right = ~left
        y_left = y[left]
        y_right = y[right]

        if len(y_left) == 0 or len(y_right) == 0:
            return 0
        
        left_entropy = DecisionTree.entropy(y_left)
        right_entropy = DecisionTree.entropy(y_right)

        weighted_entropy = (len(y_left) / len(y)) * left_entropy + (len(y_right) / len(y)) * right_entropy

        return cur_entropy - weighted_entropy

    @staticmethod
    def gini_impurity(X, y, thresh):
        # OPTIONAL
        pass

    @staticmethod
    def gini_purification(X, y, thresh):
        # OPTIONAL
        pass

    def split(self, X, y, feature_idx, thresh):
        """
        Split the dataset into two subsets, given a feature and a threshold.
        Return X_0, y_0, X_1, y_1
        where (X_0, y_0) are the subset of examples whose feature_idx-th feature
        is less than thresh, and (X_1, y_1) are the other examples.
        """
        # TODO
        left_mask = X[:, feature_idx] < thresh
        right_mask = ~left_mask 
        X_left = X[left_mask]
        y_left = y[left_mask]
        X_right = X[right_mask]
        y_right = y[right_mask]
        return X_left, y_left, X_right, y_right

    def fit(self, X, y):
        # TODO
        self.labels = y

        if self.max_depth == 0 or len(np.unique(y)) == 1:
            self.pred = Counter(y).most_common(1)[0][0]
            return

        best_info_gain = -np.inf
        best_feature = None
        best_thresh = None
        best_splits = None


        for i in range(X.shape[1]):

            values = np.unique(X[:, i])
            if len(values) <= 1:
                continue
            for j in range(len(values) - 1): 
                thresh = (values[j] + values[j + 1]) / 2.0
                ig = DecisionTree.information_gain(X[:, i], y, thresh)
                if ig > best_info_gain:
                    best_info_gain = ig
                    best_feature = i
                    best_thresh = thresh
                    best_splits = self.split(X, y, i, thresh)

        if best_info_gain <= 0 or best_splits is None:
            self.pred = Counter(y).most_common(1)[0][0]
            return

        self.split_idx = best_feature
        self.thresh = best_thresh

        X_left, y_left, X_right, y_right = best_splits
        self.left = DecisionTree(max_depth=self.max_depth - 1, feature_labels=self.features)
        self.left.fit(X_left, y_left)
        self.right = DecisionTree(max_depth=self.max_depth - 1, feature_labels=self.features)
        self.right.fit(X_right, y_right)

    def predict(self, X):
        # TODO
        predictions = [self._predict_row(x) for x in X]
        return np.array(predictions)
    
    def _predict_row(self, x):
        if self.pred is not None:
            return self.pred
        if x[self.split_idx] < self.thresh:
            return self.left._predict_row(x)
        else:
            return self.right._predict_row(x)
    
    def _to_graphviz(self, node_id):
        # If this node is a leaf, indicated by self.pred being set, then just show the prediction.
        if self.pred is not None:
            return f'{node_id} [label="Prediction: {self.pred}\nSamples: {len(self.labels)}"];\n'
        else:
            graph = f'{node_id} [label="{self.features[self.split_idx]} < {self.thresh:.2f}"];\n'
            left_id = node_id * 2 + 1
            right_id = node_id * 2 + 2
            if self.left is not None:
                graph += f'{node_id} -> {left_id};\n'
                graph += self.left._to_graphviz(left_id)
            if self.right is not None:
                graph += f'{node_id} -> {right_id};\n'
                graph += self.right._to_graphviz(right_id)
            return graph


    def to_graphviz(self):
        graph = "digraph Tree {\nnode [shape=box];\n"
        graph += self._to_graphviz(0)
        graph += "}\n"
        return graph
        
    def __repr__(self):
        # If this node is a leaf, self.pred will be set.
        if self.pred is not None:
            return "%s (%s)" % (self.pred, len(self.labels))
        else:
            return "[%s < %s: %s | %s]" % (self.features[self.split_idx],
                                        self.thresh, self.left.__repr__(),
                                        self.right.__repr__())

class BaggedTrees(BaseEstimator, ClassifierMixin):

    def __init__(self, params=None, n=200):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [
            DecisionTreeClassifier(random_state=i, **self.params)
            for i in range(self.n)
        ]

    def fit(self, X, y):
        # TODO
        n_samples = X.shape[0]
        for tree in self.decision_trees:
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            tree.fit(X_bootstrap, y_bootstrap)

    def predict(self, X):
        # TODO
        all_preds = np.array([tree.predict(X) for tree in self.decision_trees])
        all_preds = all_preds.T  
        majority_votes = []
        for row in all_preds:
            majority_votes.append(Counter(row).most_common(1)[0][0])
        return np.array(majority_votes)



class RandomForest(BaggedTrees):

    def __init__(self, params=None, n=200, m=1):
        if params is None:
            params = {}
        params['max_features'] = m
        self.m = m
        super().__init__(params=params, n=n)


class BoostedRandomForest(RandomForest):
    # OPTIONAL
    def fit(self, X, y):
        # OPTIONAL
        pass
    
    def predict(self, X):
        # OPTIONAL
        pass



def preprocess(data, fill_mode=True, min_freq=10, onehot_cols=[]):
    # Convert everything to strings first
    data = np.array(data, dtype=str)
    
    # Replace empty strings with the placeholder '-1'
    data[data == ''] = '-1'
    
    # One-hot encode specified columns
    onehot_encoding = []
    onehot_features = []
    for col in onehot_cols:
        counter = Counter(data[:, col])
        for term, freq in counter.most_common():
            if term == '-1':  # skip missing values
                continue
            if freq <= min_freq:
                break
            onehot_features.append(term)
            onehot_encoding.append((data[:, col] == term).astype(float))
        data[:, col] = '0'
    
    # Convert onehot_encoding list to a numpy array and transpose it.
    if len(onehot_encoding) > 0:
        onehot_encoding = np.array(onehot_encoding).T
    else:
        onehot_encoding = np.empty((data.shape[0], 0))
    
    # Now convert the main data to floats.
    data_numeric = np.array(data, dtype=float)
    
    # Replace missing data (marked as -1) with the mode of each column.
    if fill_mode:
        for i in range(data_numeric.shape[1]):
            # Identify missing entries: values equal to -1 (using a tolerance)
            missing_mask = np.abs(data_numeric[:, i] + 1) < eps
            non_missing = data_numeric[~missing_mask, i]
            if non_missing.size == 0:
                continue  # if entire column is missing, skip it
            # Ensure non_missing is a numpy array (it should be already, but just to be safe)
            non_missing = np.array(non_missing)
            # Handle the case where non_missing is a scalar or contains only one value.
            if non_missing.ndim == 0 or non_missing.size == 1:
                mode_val = non_missing.item()
            else:
                mode_result = stats.mode(non_missing, nan_policy='omit')
                mode_result = stats.mode(non_missing, nan_policy='omit')
                mode_val = mode_result.mode
                mode_val = mode_val[0] if isinstance(mode_val, np.ndarray) else mode_val

            # Replace missing entries with the computed mode.
            data_numeric[missing_mask, i] = mode_val
    
    final_data = np.hstack([data_numeric, onehot_encoding])
    
    return final_data, onehot_features





def evaluate(clf):
    print("Cross validation", cross_val_score(clf, X, y))
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        first_splits = [
            (features[term[0]], term[1]) for term in counter.most_common()
        ]
        print("First splits", first_splits)


def generate_submission(testing_data, predictions, dataset="titanic"):
    assert dataset in ["titanic", "spam"], f"dataset should be either 'titanic' or 'spam'"
    # This code below will generate the predictions.csv file.
    if isinstance(predictions, np.ndarray):
        predictions = predictions.astype(int)
    else:
        predictions = np.array(predictions, dtype=int)
    assert predictions.shape == (len(testing_data),), "Predictions were not the correct shape"
    df = pd.DataFrame({'Category': predictions})
    df.index += 1  # Ensures that the index starts at 1.
    df.to_csv(f'predictions_{dataset}.csv', index_label='Id')



if __name__ == "__main__":
    dataset = "spam"
    # dataset = "spam"
    params = {
        "max_depth": 5,
        # "random_state": 6,
        "min_samples_leaf": 10,
    }
    N = 100

    if dataset == "titanic":
        # Load titanic data
        path_train = 'datasets/titanic/titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None)
        path_test = 'datasets/titanic/titanic_testing_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=None)
        y = data[1:, 0]  # label = survived
        class_names = ["Died", "Survived"]

        labeled_idx = np.delete(np.arange(len(y)), 705)
        y = np.array(y[labeled_idx], dtype=float).astype(int)
        print("\n\nPart (b): preprocessing the titanic dataset")
        X, onehot_features = preprocess(data[1:, 1:], onehot_cols=[1, 5, 7, 8])
        X = X[labeled_idx, :]
        Z, _ = preprocess(test_data[1:, :], onehot_cols=[1, 5, 7, 8])
        assert X.shape[1] == Z.shape[1]
        features = list(data[0, 1:]) + onehot_features

    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription",
            "creative", "height", "featured", "differ", "width", "other",
            "energy", "business", "message", "volumes", "revision", "path",
            "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
            "square_bracket", "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = 'datasets/spam_data/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

    print("Features", features)
    print("Train/test size", X.shape, Z.shape)

    # Decision Tree
    print("\n\nDecision Tree")
    dt = DecisionTree(max_depth=3, feature_labels=features)
    dt.fit(X, y)

    # Visualize Decision Tree
    print("\n\nTree Structure")
    # Print using repr
    print(dt.__repr__())
    # Save tree to pdf
    graph_from_dot_data(dt.to_graphviz())[0].write_pdf("%s-basic-tree.pdf" % dataset)

    # Random Forest
    print("\n\nRandom Forest")
    rf = RandomForest(params, n=N, m=np.int_(np.sqrt(X.shape[1])))
    rf.fit(X, y)
    evaluate(rf)

    # Generate Test Predictions
    print("\n\nGenerate Test Predictions")
    pred = rf.predict(Z)
    generate_submission(Z, pred, dataset)