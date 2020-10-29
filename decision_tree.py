#Michael Rao
#1001558150

import os
import sys
from math import pi
from math import exp
import array
import random
import numpy as np


# Load file
def load_file(file_name):
    dataset = list()
    with open(file_name, 'r') as file:
        lines = file.readlines()
        for line in lines:
            dataset.append(line.split())
    return dataset

# Convert string column to float
def create_floats(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

class Node:
    def __init__(self, result):
        self.result = result
        self.left_child = None 
        self.right_child = None

# Convert string to int
def create_int(dataset, column):
    class_val = [row[column] for row in dataset]
    unique = set(class_val)
    lookup = dict()
    class_lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = int(value)
        class_lookup[value] = 0
    for row in dataset:
        row[column] = lookup[row[column]]
        class_lookup[row[column]] += 1
    return class_lookup


def DISTRIBUTION(dataset, class_lookup):
    dist = dict()
    total = 0
    for row in dataset:
        total += 1
        if row[-1] not in dist:
            dist[row[-1]] = 1
        else:
            dist[row[-1]] += 1
    for key in dist:
        dist[key] = (dist[key]/total) 
    return dist

# Information gain
def INFORMATION_GAIN(examples, A, threshold, class_lookup):
    # entropy at node - weighted average of entropies at sub-nodes
    examples_right = list()
    examples_left = list()
    entropy = [0,0,0]
    for row in examples:
        if row[A] < threshold:
            examples_left.append(row)
        else:
            examples_right.append(row)
    entropy[0] = ENTROPY(examples,A, class_lookup)
    entropy[1] = ENTROPY(examples_left,A, class_lookup)
    entropy[2] = ENTROPY(examples_right,A, class_lookup)
    info_gain = entropy[0] - ((len(examples_left)/len(examples))*entropy[1]) - ((len(examples_right)/len(examples))*entropy[2])
    return info_gain

# Entropy for node
def ENTROPY(examples,A, class_lookup):
    ex_distrobution = DISTRIBUTION(examples, class_lookup)
    entropy = 0
    for num in ex_distrobution:
        if num == 0:
            entropy = entropy - 0
        else:
            entropy = entropy - (ex_distrobution[num]*np.log2(ex_distrobution[num]))
    return entropy

# Select column 
def SELECT_COLUMN(examples, A):
    att = list()
    for row in examples:
        att.append(row[A])
    return att

# Optimized
def CHOOSE_ATTRIBUTE(examples, attributes, class_lookup):
    max_gain = best_attribute = best_threshold = -1
    for A in attributes:
        attribute_values = SELECT_COLUMN(examples, A)
        L = min(attribute_values)
        M = max(attribute_values)
        for K in range(50):
            threshold = L + K * (M - L)/51
            gain = INFORMATION_GAIN(examples, A, threshold, class_lookup)
            if gain > max_gain:
                max_gain = gain
                best_attribute = A 
                best_threshold = threshold
    return (best_attribute, best_threshold, max_gain)

# Random 
def CHOOSE_ATTRIBUTE_RANDOM(examples, attributes, class_lookup):
    max_gain = best_attribute = best_threshold = -1
    A = attributes[np.random.randint(0, len(attributes))]
    attribute_values = SELECT_COLUMN(examples, A)
    L = min(attribute_values)
    M = max(attribute_values)
    for K in range(50):
        threshold = L + K * (M - L)/51
        gain = INFORMATION_GAIN(examples, A, threshold, class_lookup)
        if gain > max_gain:
            max_gain = gain
            best_attribute = A
            best_threshold = threshold
    return (best_attribute, best_threshold, gain)

#All same values of class
def CHECK_CLASS(examples):
    target = len(examples[0]) - 1
    temp = examples[0][target]
    ret_val = False
    if all(e[target] == temp for e in examples):
        ret_val = True
    return ret_val

# Decision Tree
def DTL(examples, attributes, default, pruning, class_lookup, option):
    if len(examples) < pruning or len(examples) == 0:
        return Node(default)
    elif CHECK_CLASS(examples):
        return Node(DISTRIBUTION(examples, class_lookup))
    else:
        if option == "optimized":
            best_attribute, best_threshold, gain = CHOOSE_ATTRIBUTE(examples, attributes, class_lookup)
        elif option == "randomized":
            best_attribute, best_threshold, gain = CHOOSE_ATTRIBUTE_RANDOM(examples, attributes, class_lookup)
        tree = Node((best_attribute, best_threshold, gain))
        examples_right = list()
        examples_left = list()
        for row in examples:
            if row[best_attribute] < best_threshold:
                examples_left.append(row)
            else:
                examples_right.append(row)
        dist = DISTRIBUTION(examples, class_lookup)
        
        tree.left_child = DTL(examples_left, attributes, dist, pruning, class_lookup, option)
        tree.right_child = DTL(examples_right, attributes, dist, pruning, class_lookup, option)
        return tree

def levelOrder(root):
    ret = []
    if not root:
        return ret
    q = [root]
    
    while q:
        ql = len(q)
        levelList = []
        while ql:
            ql -= 1
            node = q.pop(0)
            if node.left_child:
                q.append(node.left_child)
            if node.right_child:
                q.append(node.right_child)
            levelList.append(node.result)
            
        ret.append(levelList)
    return ret

def test(row, root, results):
    if root.left_child is None and root.right_child is None:
        results.append(root.result)
        return 
    if row[root.result[0]] < root.result[1]:
        test(row, root.left_child, results)
    else:
        test(row, root.right_child, results)

def decision_tree():
    if(len(sys.argv) < 5):
        print("Insufficient command line args")
        exit()

    forest = list()
    training_file = load_file(sys.argv[1])
    test_file = load_file(sys.argv[2])
    option = sys.argv[3]
    pruning = int(sys.argv[4])

    for i in range(len(training_file[0])):
        create_floats(training_file,i)

    for i in range(len(test_file[0])):
        create_floats(test_file,i)

    class_lookup = create_int(training_file, len(training_file[0]) - 1)
    test_lookup = create_int(test_file, len(test_file[0]) - 1)
    attributes = np.arange(len(training_file[0]) - 1)
    default = DISTRIBUTION(training_file, class_lookup)
    if option == 'forest3':
        for i in range(3):
            forest.append(DTL(training_file, attributes, default, pruning, class_lookup, "randomized"))
    elif option == 'forest15':
        for i in range(15):
            forest.append(DTL(training_file, attributes, default, pruning, class_lookup, "randomized"))
    else:
        
        forest.append(DTL(training_file, attributes, default, pruning, class_lookup, option))

    tree_num = 1
    node_id = 0
    for i in range(len(forest)):
        tree_lvl =  levelOrder(forest[i])
        node_id = 1
        for row in tree_lvl:
            for elem in row:
                if isinstance(elem,dict):
                    print('tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f' % (tree_num, node_id, -1, -1, 0))    
                else:
                    print('tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f' % (tree_num, node_id, elem[0], elem[1], elem[2]))   
                node_id = node_id + 1
        tree_num = tree_num + 1
    
    final_results = list()
    for i in range(len(test_file)):
        results = list()
        for j in range(len(forest)):
            temp_result = list()
            test(test_file[i], forest[j],temp_result)
            if not results:
                results = temp_result
            else:
                maxTemp = max(temp_result[0], key=temp_result[0].get)
                maxRes = max(results[0], key=results[0].get)
                if temp_result[0][maxTemp] > results[0][maxRes]:
                    results = temp_result
        final_results.append(results)

    total = 0
    for k in range(len(final_results)):
        acc = 0
        maxKey = max(final_results[k][0], key=final_results[k][0].get)
        if maxKey == test_file[k][-1]:
            acc = 1
            total += 1
        print('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f' % (k+ 1, maxKey, test_file[k][-1], acc))

    print(float(total))
    print(k)
    print('classification accuracy=%6.4f\n' % ( float(total) / k))

    

if __name__ == '__main__':
    decision_tree()