from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import math
from collections import Counter
import numpy as np

iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=123)

def entropy_func(class_count, num_samples):
    return sum([-class_count/num_samples * math.log(class_count / num_samples, 2) if class_count else 0])


class Group:
    def __init__(self, group_classes):
        self.group_classes = group_classes
        self.entropy = self.group_entropy()

    def __len__(self):
        return self.group_classes.size

    def group_entropy(self):
        pass


class Node:
    def __init__(self, split_feature, split_val, depth=None, child_node_a=None, child_node_b=None, val=None):
        self.split_feature = split_feature
        self.split_val = split_val
        self.depth = depth
        self.child_node_a = child_node_a
        self.child_node_b = child_node_b
        self.val = val

    def predict(self, data):
        pass


class DecisionTreeClassifier(object):
    def __init__(self, max_depth):
        self.depth = 0
        self.max_depth = max_depth
        self.tree = None

    @staticmethod
    def get_split_entropy(group_a, group_b):
        pass

    def get_information_gain(self, parent_group, child_group_a, child_group_b):
        pass

    def get_best_feature_split(self, feature_values, classes):
        pass

    def get_best_split(self, data, classes):
        pass

    def build_tree(self, data, classes, depth=0):
        pass

    def predict(self, data):
        return self.tree.predict(data)