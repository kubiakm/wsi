from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import math
from collections import Counter
import numpy as np

iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=123)

# funkcja entropii
def entropy_func(class_count, num_samples:list):
    total_num_samples = sum(num_samples)
    entropy = sum([-num_samples[count]/total_num_samples* math.log(num_samples[count]/total_num_samples, 2) if count else 0 for count in range(class_count)])
    return entropy

# zbiór klas, jako parametr przyjmuje listę klas, zwraca jej długość i entropię zbioru
class Group:
    def __init__(self, group_classes:list):
        self.group_classes = group_classes
        self.entropy = self.group_entropy()
        self.size = len(group_classes) 

    def __len__(self):
        return len(self.group_classes)

 # oblicza entropię na podstawie liczby klas, liczby unikalnych klas i tego ile razy wystąpiły
    def group_entropy(self):
        unique_classes = list(np.unique(self.group_classes))
        class_count = len(unique_classes)
        num_samples = []
        for x in unique_classes:
            samples = np.count_nonzero(self.group_classes == x)
            num_samples.append(samples)
        return entropy_func(class_count, num_samples)

# węzeł, przyjmuje indeks atrybutu, wartość względem której następuje podział na lewą i prawą klasę, głębokość w drzewie, węzły dzieci (lewego i prawego) i wartość liścia
class Node:
    def __init__(self, split_feature, split_val, depth=None, child_node_a=None, child_node_b=None, val=None):
        self.split_feature = split_feature
        self.split_val = split_val
        self.depth = depth
        self.child_node_a = child_node_a
        self.child_node_b = child_node_b
        self.val = val

    # zwraca watość decyzyjną danego przykładu
    def predict(self, data):
        if self.val is None:
            if data[self.split_feature] <= self.split_val:
                return self.child_node_a.predict(data)
            else:
                return self.child_node_b.predict(data)
        else:
            return self.val
# drzewo decyzyjne, jako parametry przyjmuje maksymalną głębokość, tworzone na podstawie pierwszego węzła stanowiącego atrybut tree
class DecisionTreeClassifier(object):
    def __init__(self, max_depth):
        self.depth = 0
        self.max_depth = max_depth
        self.tree = None

    # zwraca entropię każdego ze zbiorów
    @staticmethod
    def get_split_entropy(group_a, group_b):
        split_entropy_a = group_a.entropy
        split_entropy_b = group_b.entropy
        return split_entropy_a, split_entropy_b
         
    # informacja, którą zyskujemy dzięki klasyfikacji rodzica na dwójkę dzieci 
    def get_information_gain(self, parent_group, child_group_a, child_group_b):
        parent_entropy = parent_group.entropy
        child_a_entropy, child_b_entropy = self.get_split_entropy(child_group_a, child_group_b)
        information_gain = parent_entropy - child_a_entropy*(child_group_a.size/parent_group.size) -child_b_entropy*(child_group_b.size/parent_group.size)
        return information_gain

    # dzieli na lewą (a) i prawą (b) klasę względem jakieś wartości granicznej podziału, dzieli dane według tej wartości
    # tworzy obiekty rodzica
    # zwraca uzyskaną informację oraz klasy lewego i prawego dziecka wraz danymi
    def get_left_right_split(self, data, feature_values, split_value, classes):
        a_classes = [classes[n] for n in range(len(classes)) if feature_values[n] <= split_value]
        b_classes = [classes[n] for n in range(len(classes)) if feature_values[n] > split_value]
        a_data = [data[n] for n in range(len(classes)) if feature_values[n] <= split_value]
        b_data = [data[n] for n in range(len(classes)) if feature_values[n] > split_value]
        parent_group = Group(classes)
        a_group = Group(a_classes)
        b_group = Group(b_classes)
        info_gain = self.get_information_gain(parent_group, a_group, b_group)
        return info_gain, a_classes, b_classes, a_data, b_data

    # najlepszy atrybut decyzyjny, zwraca uzyskaną informację, wartość atrybutu, klasy lewą i prawą wraz z danymi do nich należącymi
    def get_best_feature_split(self, data, feature_values, classes):
        best_info_gain = 0
        best_split_value = None
        best_a_classes = None
        best_b_classes = None
        best_a_data = None
        best_b_data = None
        for split_value in feature_values:
            info_gain, a_classes, b_classes, a_data, b_data = self.get_left_right_split(data, feature_values, split_value, classes)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_split_value = split_value
                best_a_classes = a_classes
                best_b_classes = b_classes
                best_a_data = a_data
                best_b_data = b_data
                return best_info_gain, best_split_value, best_a_classes, best_b_classes, best_a_data, best_b_data

    # najlepszy podział pod względem wszystkich atrybutów i ich wartości, zwraca zyskaną informację, najlepszy atrybut, jego wartość, klasy i dane lewej oraz prawej strony
    def get_best_split(self, data, classes):
        converted_data = [None]*len(data[0])
        for x in range(len(data[0])):
          feature_values = [data[i][x] for i in range(len(data))]
          converted_data[x] = feature_values
        best_info_gain = 0
        best_split_feature =  None
        best_split_value =  None
        best_a_classes = None
        best_b_classes = None
        best_a_data = None
        best_b_data = None
        feature_index = 0
        for feature_vals in converted_data:
            info_gain, split_value, a_classes, b_classes, a_data, b_data = self.get_best_feature_split(data, feature_vals, classes)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_split_value = split_value
                best_split_feature = feature_index
                best_a_classes = a_classes
                best_b_classes = b_classes
                best_a_data = a_data
                best_b_data = b_data
            feature_index += 1
        return best_info_gain, best_split_value, best_split_feature, best_a_classes, best_b_classes, best_a_data, best_b_data

# buduje drzewo decyzyjne, przyjmuje jako parametry dane, zbiór klas i głębokość, zwraca węzeł lub liść
    def build_tree(self, data, classes, depth=0):
        if depth == self.max_depth or len(set(classes)) == 1:
          samples = dict(Counter(classes))
          leaf_value = max(samples, key=samples.get) # jeśli w liściu występuje kilka klas, decyzją jest klasa większościowa 
          print("NEW LEAF")
          print("LEAF CLASS VALUE")
          print(leaf_value)
          print("\n")
          return Node(None, None, None, None, None, leaf_value)
        else:
            info_gain, split_value, split_feature, a_classes, b_classes, a_data, b_data = self.get_best_split(data, classes)
            if info_gain > 0:
                a_branch = self.build_tree(a_data, a_classes, depth + 1)
                b_branch = self.build_tree(b_data, b_classes, depth + 1)
            print("NEW DECISION NODE")
            print("INFORMATION GAIN")
            print(info_gain)
            print("DECISION PARAMETER INDEX")
            print(split_feature)
            print("LIMIT VALUE")
            print(split_value)
            print("\n")
            return Node(split_feature, split_value, depth, a_branch, b_branch)


    def predict(self, data):
        return self.tree.predict(data)

decision_classifier = DecisionTreeClassifier(5)
decision_classifier.tree = decision_classifier.build_tree(x_train, y_train) # pierwszy node drzewa

predictions = []
for sample, target in zip(x_test, y_test):
    prediction = decision_classifier.predict(sample)
    predictions.append(prediction)
total_number = len(predictions)
error = 0 
for y_predicted, y in zip(predictions, y_test):
    if y_predicted != y:
        error += 1
accuracy = (total_number-error)/total_number

print("PARAMETERS")
print(x_test)
print("CLASSES")
print(y_test)
print("PREDICTIONS")
print(predictions)
print("ACCURACY")
print(accuracy)