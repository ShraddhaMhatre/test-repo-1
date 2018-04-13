
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# In[2]:


class Tree:
    def __init__(self, feature_index, midpoint, l_child, r_child, output):
        self.feature_index = feature_index
        self.midpoint = midpoint
        self.l_child = l_child
        self.r_child = r_child
        self.output = output


# In[3]:


def entropy(target):
    result = 0
    types, counts = np.unique(target, return_counts=True)
    freqs = counts.astype('float')/len(target)
    for p in freqs:
        if p != 0.0:
            result -= p * np.log2(p)
    return result


# In[4]:


def calcThreshold(feature, target):
    candidates = []
    sorted_feature = [feature for feature,target in sorted(zip(feature,target))]
    sorted_target = [target for feature,target in sorted(zip(feature,target))]
    sorted_f = np.array(sorted_feature)
    sorted_t = np.array(sorted_target)
    change = np.where(sorted_t[:-1] != sorted_t[1:])[0]
    for i in change:
        midpoint = float(sorted_f[i-1]) + (float(sorted_f[i]) - float(sorted_f[i-1])) / 2
        candidates.append(midpoint)
    return candidates


# In[5]:


def splitDataset(feature_index, midpoint, train_data):
    return split(train_data, train_data[:,feature_index].astype(float) < midpoint)


# In[6]:


def split(arr, cond):
    return [arr[cond], arr[~cond]]


# In[7]:


def calcIg(l_counts, r_counts, Hl, Hr, Hq):
    sum_l_count = np.sum(l_counts)
    sum_r_count = np.sum(r_counts)
    total = sum_l_count + sum_r_count
    return (Hq - ((sum_l_count / total) * Hl + (sum_r_count / total) * Hr))


# In[8]:


def buildTree(train_data, Hq):
    num_of_instances = np.size(train_data[:, :-1], 0)
    if(np.size(train_data[:, :-1],0) == 0 or np.size(train_data[:,-1]) == 0):
        return None
    elif(Hq == 0):
        return Tree(None, None, None, None, np.unique(train_data[:,-1])[0])
    elif(num_of_instances < n_min * initial_training_size):
        classes, counts = np.unique(train_data[:,-1], return_counts=True)
        return Tree(None, None, None, None, classes[np.argmax(counts)])
    else:
        max_Ig = 0.0

        for f in range(np.size(train_data[:, :-1], 1)):
            midpoints = calcThreshold(train_data[:, :-1][:,f], train_data[:,-1])
            for m in midpoints:
                left_dataset, right_dataset = splitDataset(f, m, train_data)
                l_classes, l_counts = np.unique(left_dataset[:,-1], return_counts=True)
                r_classes, r_counts = np.unique(right_dataset[:,-1], return_counts=True)
                Hl = entropy(left_dataset[:,-1])
                Hr = entropy(right_dataset[:,-1])
                Ig = calcIg(l_counts, r_counts, Hl, Hr, Hq)
                if(Ig > max_Ig):
                    max_Ig = Ig
                    best_feature = f
                    midpoint = m
                    left_split = left_dataset
                    right_split = right_dataset
                    H_left = Hl
                    H_right = Hr

        if(max_Ig == 0):
            classes, counts = np.unique(train_data[:,-1], return_counts=True)
            return Tree(None, None, None, None, classes[np.argmax(counts)])

        return Tree(best_feature, 
                    midpoint, 
                    buildTree(left_split, H_left),
                    buildTree(right_split, H_right),
                    None)


# In[9]:


def getMaxOutput(classes, counts):
    return classes[np.argmax(counts)]


# In[10]:


def traverse(test_tuple, outputs, tree):
    if tree is None:
        return None
    if(tree.output is not None):
        outputs.append(tree.output)
    elif(tree.midpoint is not None and test_tuple[tree.feature_index] < tree.midpoint):
        traverse(test_tuple, outputs, tree.l_child)
    else:
        traverse(test_tuple, outputs, tree.r_child)


# In[11]:


def predictor(feature_test):
    outputs = []
    if(tree == None):
        return None
    else:
        for test_tuple in feature_test:
            traverse(test_tuple, outputs, tree)
    
    return outputs


# In[12]:


# read data from pdf
dataset = pd.read_csv("iris.csv", header=None)

# shuffle the dataset
dataset = dataset.sample(frac=1).reset_index(drop=True)

# break it into features and class
features_df = np.array(dataset.iloc[:, :-1])
class_df = np.array(dataset.iloc[:, -1])

# normalize the features
scaler = preprocessing.MinMaxScaler()
normalized_features_df = scaler.fit_transform(features_df)

# 10-fold
kf = KFold(n_splits=10)

n_mins = [0.05, 0.10, 0.15, 0.20]

avg_test_acc_arr = []
avg_train_acc_arr = []
cm = np.zeros(shape=(3,3))
for n_min in n_mins:
    print('n_min: ', n_min)
    accuracy_test_arr = []
    accuracy_train_arr = []
    print('Confusion matrix for all the 10-folds: ')
    count = 1
    for train, test in kf.split(dataset):
        print('Confusion matrix #', count)
        count += 1
        # feature_train = normalized_features_df[train]
        # class_train = class_df[train]
        # feature_test = normalized_features_df[test]
        # class_test = class_df[test]

        training_data = dataset[train]
        test_data = dataset[test]

        
        initial_entropy = entropy(class_train)

        train_y = []
        for e in class_train:
            train_y.append([e])

        init_train_data = np.append(feature_train, train_y, axis=1)

        max_Ig = 0
        initial_training_size = np.size(feature_train, 0)

        tree = buildTree(init_train_data, initial_entropy)

        test_outputs = np.array(predictor(feature_test))
        accuracy_test_arr.append(np.sum(test_outputs == class_test) * 100 / np.size(class_test, 0))
        
        train_outputs = np.array(predictor(feature_train))
        accuracy_train_arr.append(np.sum(train_outputs == class_train) * 100 / np.size(class_train, 0))
        
        cm_intermediate = confusion_matrix(class_test, test_outputs) 
        print(cm_intermediate)
        cm += cm_intermediate
    
    std_dev = np.std(accuracy_test_arr)
    avg_test_accuracy = np.mean(accuracy_test_arr)
    avg_train_accuracy = np.mean(accuracy_train_arr)
    avg_test_acc_arr.append(avg_test_accuracy)
    avg_train_acc_arr.append(avg_train_accuracy)
    print('Average Accuracy: ', avg_test_accuracy)
    print('Standard Deviation: ', std_dev)
    print(cm)

plt.scatter(n_mins, avg_test_acc_arr, c="r")
plt.scatter(n_mins, avg_train_acc_arr, c="b")
plt.ylabel("Average Accuracies")
plt.xlabel("n_mins")
plt.show()

