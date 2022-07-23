from typing import List

class DecisionTreeClassifier:

    def __init__(self, max_depth: int):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = 2

    def fit(self, X: List[List[float]], y: List[int]):
        dataset = []
        for i in range(len(X)):
            X[i].append(y[i])
            dataset.append(X[i])
        self.root = self.build_tree(dataset)

    def build_tree(self, dataset , current_depth=0):
        X = []
        Y = []
        for i in range(len(dataset)):
            temp = []
            for j in range(len(dataset[i]) -1):
                temp.append(dataset[i][j])
            X.append(temp)
            Y.append(dataset[i][len(dataset[i]) -1])
        num_samples = len(X)
        num_features = len(X[0])
        if num_samples>=self.min_samples_split and current_depth<=self.max_depth:
            best_split = self.find_best_split(dataset, num_samples, num_features)
            if best_split["info_gain"]>0:
                left_subtree = self.build_tree(best_split["dataset_left"], current_depth+1)
                right_subtree = self.build_tree(best_split["dataset_right"], current_depth+1)
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        leaf_value = self.calculate_leaf_value(Y)
        return Node(value=leaf_value)   

    def predict(self, X: List[List[float]]):
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions

    def make_prediction(self, x, tree):
        if tree.value!=None: 
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

    

    def print_tree(self, tree=None, indent=" "):
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

    def find_best_split(self, dataset, num_samples,num_features):
        best_split = {}
        max_info_gain = -float("inf")
        for feature_index in range(num_features):
            feature_values = []
            for i in range(len(dataset)):
                feature_values.append(dataset[i][feature_index])
            temp = []
            for val in feature_values:
                if not(temp.__contains__(val)):
                    temp.append(val)
            self.insertionSort(temp)
            possible_thresholds = temp
            for threshold in possible_thresholds:
                dataset_left , dataset_right = self.split_data(dataset, feature_index, threshold)
                if(len(dataset_left) > 0 and len(dataset_right) > 0):
                    y = []
                    left_y = []
                    right_y = []
                    for i in range(len(dataset)):
                        temp = len(dataset[i]) -1
                        if(temp >= 0):
                            y.append(dataset[i][len(dataset[i]) -1])
                    for i in range(len(dataset_left)):
                        temp = len(dataset_left[i]) -1
                        if(temp >= 0):
                            left_y.append(dataset_left[i][len(dataset_left[i]) -1])
                    for i in range(len(dataset_right)):
                        temp = len(dataset_right[i]) -1
                        if(temp >= 0):
                            right_y.append(dataset_right[i][len(dataset_right[i]) -1])
                    curr_info_gain = self.information_gain(y, left_y, right_y)
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
        return best_split
        
    def gini_index(self, y):
        class_labels = []
        for i in range(len(y)):
            if not(class_labels.__contains__(y[i])):
                class_labels.append(y[i])

        for step in range(1, len(class_labels)):
            key = class_labels[step]
            j = step - 1       
            while j >= 0 and key < class_labels[j]:
                class_labels[j + 1] = class_labels[j]
                j = j - 1
            class_labels[j + 1] = key
        gini = 0
        for cls in class_labels:
            ctr = 0
            for temp in y:
                if(temp == cls):
                    ctr += 1
            p_cls = ctr / len(y)
            gini += p_cls ** 2
        return 1 - gini

    def calculate_leaf_value(self, y):
        Y = list(y) 
        return max(Y, key=Y.count)

    def insertionSort(self, array):
        for step in range(1, len(array)):
            key = array[step]
            j = step - 1       
            while j >= 0 and key < array[j]:
                array[j + 1] = array[j]
                j = j - 1
            array[j + 1] = key

    def split_data(self, dataset, feature_index, threshold):
        left = []
        right = []
        for row in dataset:
            if(row[feature_index] <= threshold):
                left.append(row)
            else:
                right.append(row)
        return left, right
    
    def information_gain(self, parent, left_child, right_child):
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)
        gain = self.gini_index(parent) - (weight_left*self.gini_index(left_child) + weight_right*self.gini_index(right_child))
        return gain
   
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value

# if __name__ == '__main__':  
    # X, y = ...
    # X_train, X_test, y_train, y_test = ...

    # clf = DecisionTreeClassifier(max_depth=5)
    # clf.fit(X_train, y_train)
    # yhat = clf.predict(X_test)    
    