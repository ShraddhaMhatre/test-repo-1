
# coding: utf-8

# In[287]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import xlsxwriter


# In[288]:


class Tree:
    def __init__(self, feature_index, midpoint, l_child, r_child, output):
        self.feature_index = feature_index
        self.midpoint = midpoint
        self.l_child = l_child
        self.r_child = r_child
        self.output = output


# In[289]:


def entropy(target):
    result = 0
    types, counts = np.unique(target, return_counts=True)
    freqs = counts.astype('float')/len(target)
    for p in freqs:
        if p != 0.0:
            result -= p * np.log2(p)
    return result


# In[290]:


def calcGm(target):
    return np.sum(target) / np.size(target)


# In[291]:


def calcSSE(target, gm):
    return np.sum(np.square(np.subtract(target, gm)))


# In[292]:


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


# In[293]:


def splitDataset(feature_index, midpoint, train_data):
    return split(train_data, train_data[:,feature_index].astype(float) < midpoint)


# In[294]:


def split(arr, cond):
    return [arr[cond], arr[~cond]]


# In[295]:


def calcNode(feature, target, midpoint, feature_index, Hq, train_data):
    splitData = splitDataset(feature_index, midpoint, train_data)
    left_dataset = splitData[0]
    right_dataset = splitData[1]
    l_classes, l_counts = np.unique(left_dataset[:,-1], return_counts=True)
    r_classes, r_counts = np.unique(right_dataset[:,-1], return_counts=True)
    Hl = entropy(left_dataset[:,-1])
    Hr = entropy(right_dataset[:,-1])
    Ig = calcIg(l_counts, r_counts, Hl, Hr, Hq)
    return(left_dataset, right_dataset, l_counts, r_counts, Hl, Hr, Ig, feature_index, l_classes, r_classes, midpoint)


# In[296]:


def calcIg(l_counts, r_counts, Hl, Hr, Hq):
    sum_l_count = np.sum(l_counts)
    sum_r_count = np.sum(r_counts)
    total = sum_l_count + sum_r_count
    return (Hq - ((sum_l_count / total) * Hl + (sum_r_count / total) * Hr))

# calcIg([3,4], [3,2], 0.9852, 0.9710, 1)
# calcIg([5,4], [1,2], 0.9911, 0.9183, 1)


# In[297]:


def calcBestNode(feature, c_train, Hq, train_data):
    midpoints = calcThreshold(feature, c_train)
    return calcNode(feature, c_train, midpoints[0], 0, Hq, train_data)

def buildTree(train_data, sse):
    num_of_instances = np.size(train_data[:, :-1], 0)
    if(np.size(train_data[:, :-1],0) == 0 or np.size(train_data[:,-1]) == 0):
        return None
    elif(sse == 0):
        return Tree(None, None, None, None, np.average(train_data[:,-1]))
    elif(num_of_instances < n_min * initial_training_size):
        return Tree(None, None, None, None, np.average(train_data[:,-1]))
    else:
        max_diff = 0.0

        for f in range(np.size(train_data[:, :-1], 1)):
#             midpoints = (train_data[:,f][1:] + train_data[:,f][:-1]) / 2
            midpoints = calcThreshold(train_data[:, :-1][:,f], train_data[:,-1])
            for m in midpoints:
                left_dataset, right_dataset = splitDataset(f, m, train_data)
                l_sse = calcSSE(left_dataset[:,-1], calcGm(left_dataset[:,-1]))
                r_sse = calcSSE(right_dataset[:,-1], calcGm(right_dataset[:,-1]))
                diff = sse - (np.size(left_dataset[:,-1]) / num_of_instances) * l_sse 
                - (np.size(right_dataset[:,-1]) / num_of_instances) * r_sse
                if(diff > max_diff):
                    best_feature = f
                    midpoint = m
                    left_split = left_dataset
                    right_split = right_dataset
                    sse_left = l_sse
                    sse_right = r_sse

        if(max_diff == 0):
            classes, counts = np.unique(train_data[:,-1], return_counts=True)
            return Tree(None, None, None, None, np.average(train_data[:,-1]))

        return Tree(best_feature, 
                    midpoint, 
                    buildTree(left_split, sse_left),
                    buildTree(right_split, sse_right),
                    None)


# In[298]:


def getMaxOutput(classes, counts):
    return classes[np.argmax(counts)]


# In[299]:


def traverse(test_tuple, outputs, tree):
    if tree is None:
        return None
    if(tree.output is not None):
        outputs.append(tree.output)
    elif(tree.midpoint is not None and test_tuple[tree.feature_index] < tree.midpoint):
        traverse(test_tuple, outputs, tree.l_child)
    else:
        traverse(test_tuple, outputs, tree.r_child)


# In[300]:


def predictor(feature_test):
    outputs = []
    if(tree == None):
        return None
    else:
        for test_tuple in feature_test:
            traverse(test_tuple, outputs, tree)
    
    return outputs


# In[301]:


# read data from pdf
dataset = pd.read_csv("housing.csv", header=None)
dataset = dataset.sample(frac=1).reset_index(drop=True)

# break it into features and class
features_df = np.array(dataset.iloc[:, :-1])
class_df = np.array(dataset.iloc[:, -1])

# normalize the features
scaler = preprocessing.MinMaxScaler()
normalized_features_df = scaler.fit_transform(features_df)

#10-fold
kf = KFold(n_splits=10)

n_mins = [0.05, 0.10, 0.15, 0.20]

# workbook = xlsxwriter.Workbook('q1.xlsx')    
# worksheet = workbook.add_worksheet('1.1-iris')
# row = 1
# col = 0
# worksheet.write(0, 0, 'Nmin')
# worksheet.write(0, 1, 'Avg_Accuracy')
# worksheet.write(0, 2, 'Std_Deviation')
avg_test_sse_arr = []
avg_train_sse_arr = []
cm = np.zeros(shape=(3,3))
for n_min in n_mins:
    print('n_min: ', n_min)
    sse_test_arr = []
    sse_train_arr = []
    for train, test in kf.split(dataset):
        feature_train = normalized_features_df[train]
        class_train = class_df[train]
        feature_test = normalized_features_df[test]
        class_test = class_df[test]
        
        initial_sse = calcSSE(class_train, calcGm(class_train))

        train_y = []
        for e in class_train:
            train_y.append([e])

        init_train_data = np.append(feature_train, train_y, axis=1)

        max_Ig = 0
        initial_training_size = np.size(feature_train, 0)

        tree = buildTree(init_train_data, initial_sse)

        test_outputs = np.array(predictor(feature_test))
        sse_test_arr.append(np.sum(np.square(np.subtract(test_outputs, class_test))))
#         accuracy_test_arr.append(np.sum(test_outputs == class_test) * 100 / np.size(class_test, 0))
        
        train_outputs = np.array(predictor(feature_train))
        sse_train_arr.append(np.sum(np.square(np.subtract(train_outputs, class_train))))
#         accuracy_train_arr.append(np.sum(train_outputs == class_train) * 100 / np.size(class_train, 0))
        
#         cm += confusion_matrix(class_test, test_outputs)
    std_dev = np.std(sse_test_arr)
    avg_test_sse = np.mean(sse_test_arr)
    avg_train_sse = np.mean(sse_train_arr)
    avg_test_sse_arr.append(avg_test_sse)
    avg_train_sse_arr.append(avg_train_sse)
    
    print('MSE: ', avg_test_sse)
    print('Standard Deviation: ', std_dev)
    
#     print(cm)
#     worksheet.write(row, col, n_min)
#     worksheet.write(row, col + 1, avg_accuracy)
#     worksheet.write(row, col + 2, std_dav)
#     row += 1

plt.scatter(n_mins, avg_test_sse_arr, c="r")
plt.scatter(n_mins, avg_train_sse_arr, c="b")
plt.ylabel("MSE")
plt.xlabel("n_mins")
plt.show()
# fo = open("q1.2-a.txt", "w+")
# fo.write('n_min: ' + '0.05\n\n')
# fo.write(str(cm) + '\n\n')
# fo.write('Average Accuracy: ' + str(avg_accuracy) + '\n')
# fo.write('Standard Deviation: ' + str(std_dev) + '\n')
# fo.close()
# workbook.close()

