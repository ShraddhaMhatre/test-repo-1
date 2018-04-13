
# coding: utf-8

# In[88]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
import xlsxwriter


# In[89]:


class Tree:
    def __init__(self, feature_index, feature_values, child_list, output):
        self.feature_index = feature_index
        self.feature_values = feature_values
        self.child_list = child_list
        self.output = output


# In[90]:


def entropy(target):
    result = 0
    types, counts = np.unique(target, return_counts=True)
    freqs = counts.astype('float')/len(target)
    for p in freqs:
        if p != 0.0:
            result -= p * np.log2(p)
    return result


# In[91]:


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


# In[92]:


def splitDataset(feature_index, train_data):
    possible_outcomes = np.unique(train_data[:, feature_index])
    result = []
    for po in possible_outcomes:
        result.append(train_data[train_data[:, feature_index] == po])
    
    return result


# In[93]:


def calcIg(counts, H, Hq):
    total = 0
    sums = []
    intermediate = 0
    
    for count in counts:
        sum_count = np.sum(count)
        sums.append(sum_count)
        total += sum_count
    
    for count, h in zip(sums, H):
        intermediate += count / total * h
        
    return (Hq - intermediate)


# In[94]:


def getMaxOutput(classes, counts):
    return classes[np.argmax(counts)]


# In[95]:


def buildTree(train_data, Hq):
    num_of_instances = np.size(train_data, 0)
    if(num_of_instances == 0):
        return None
    elif(Hq == 0):
        return Tree(None, None, None, np.unique(train_data[:,-1])[0])
    elif(num_of_instances < n_min * initial_training_size):
        classes, counts = np.unique(train_data[:,-1], return_counts=True)
        return Tree(None, None, None, classes[np.argmax(counts)])
    else:
        max_Ig = 0.0
        for f in range(np.size(train_data[:, :-1], 1)):
            split_data = splitDataset(f, train_data)
            
            entropy_arr = []
            classes_arr= []
            counts_arr = []
            
            for data in split_data:
                classes, counts = np.unique(data[:,-1], return_counts=True)
                H = entropy(data[:,-1])
                classes_arr.append(classes)
                counts_arr.append(counts)
                entropy_arr.append(H)
            
            Ig = calcIg(counts_arr, entropy_arr, Hq)   
                
#                 print('inner Ig: ', Ig)
            if(Ig > max_Ig):
#                     print('Ig: ', Ig)
#                     print('max_Ig: ', max_Ig)
                max_Ig = Ig
#                     print('updated max_Ig: ', max_Ig)
                best_feature = f
                feature_values = np.unique(train_data[:, f]).tolist()
#                     print('best_feature: ', best_feature)
                split_list = split_data
#                     print('split_size0: ', np.size(split_data[0], 0))
#                     print('split_size1: ', np.size(split_data[1], 0))
                H_arr = entropy_arr

        if(max_Ig == 0):
            classes, counts = np.unique(train_data[:,-1], return_counts=True)
            return Tree(None, None, None, classes[np.argmax(counts)])

        return Tree(best_feature,
                    feature_values,
                    callBuildTree(split_list, H_arr),
                    None)


# In[96]:


def callBuildTree(split_list, H_arr):
    tree = []
    for train_data, H in zip(split_list, H_arr):
#         print(np.size(train_data, 0))
        tree.append(buildTree(train_data, H))
    return tree


# In[97]:


def traverse(test_tuple, outputs, tree):
    if tree is None:
        return None
    if(tree.output is not None):
        outputs.append(tree.output)
    else:
#         traverse(test_tuple, outputs, tree.child_list[np.where(tree.feature_values == test_tuple[tree.feature_index])[0]])
        traverse(test_tuple, outputs, tree.child_list[tree.feature_values.index(test_tuple[tree.feature_index])]) 


# In[98]:


def predictor(feature_test):
    outputs = []
    if(tree == None):
        return None
    else:
        for test_tuple in feature_test:
            traverse(test_tuple, outputs, tree)
    
    return outputs


# In[99]:


# read data from pdf
dataset = pd.read_csv("mushroom.csv", header=None)
# dataset = dataset.sample(frac=1).reset_index(drop=True)

# # break it into features and class
features_df = np.array(dataset.iloc[:, :-1])
class_df = np.array(dataset.iloc[:, -1])

# # normalize the features
# scaler = preprocessing.MinMaxScaler()
# normalized_features_df = scaler.fit_transform(features_df)

# 10-fold
kf = KFold(n_splits=10)

n_mins = [0.05, 0.10, 0.15]

# workbook = xlsxwriter.Workbook('q1.xlsx')    
# worksheet = workbook.add_worksheet('1.1-a-iris')
# row = 1
# col = 0
# worksheet.write(0, 0, 'Nmin')
# worksheet.write(0, 1, 'Avg_Accuracy')
# worksheet.write(0, 2, 'Std_Deviation')
# count = 0;
for n_min in n_mins:
    print(n_min)
    accuracy_arr = []
    for train, test in kf.split(dataset):
        feature_train = features_df[train]
        class_train = class_df[train]
        feature_test = features_df[test]
        class_test = class_df[test]
        initial_entropy = entropy(class_train)

        train_y = []
        for e in class_train:
            train_y.append([e])

        init_train_data = np.append(feature_train, train_y, axis=1)

        max_Ig = 0
        initial_training_size = np.size(feature_train, 0)

        tree = buildTree(init_train_data, initial_entropy)

        outputs = np.array(predictor(feature_test))
        accuracy_arr.append(np.sum(outputs == class_test) * 100 / np.size(class_test, 0))
    
    std_dav = np.std(accuracy_arr)
    avg_accuracy = np.mean(accuracy_arr)
    print('std_dav: ', std_dav)
    print('avg_acc: ', avg_accuracy)
#     worksheet.write(row, col, n_min)
#     worksheet.write(row, col + 1, avg_accuracy)
#     worksheet.write(row, col + 2, std_dav)
#     row += 1

# workbook.close()

