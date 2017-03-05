from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import ShuffleSplit
import numpy as np

def trainModel(features, target):
    plength = features[:, 2]
    # use numpy operations to get setosa features
    is_setosa = (target == 0)
    # This is the important step:
    max_setosa =plength[is_setosa].max()
    min_non_setosa = plength[~is_setosa].min()
    print('Maximum of setosa: {0}.'.format(max_setosa))
    print('Minimum of others: {0}.'.format(min_non_setosa))
    
    features = features[~is_setosa]
    target = target[~is_setosa]
    virginica = (target == 2)
    
    best_acc = -1.0
    for fi in range(features.shape[1]):
    # We are going to generate all possible threshold for this feature
        thresh = features[:,fi].copy()
        thresh.sort()
        # Now test all thresholds:
        for t in thresh:
            pred = (features[:,fi] > t)
            acc = (pred == virginica).mean()
            if acc > best_acc:
                best_acc = acc
                best_fi = fi
                best_t = t
    print('Virginica versus Versicolor classification:')
    print('Best accuracy: {0}'.format(best_acc))
    print('Best feature: {0}'.format(feature_names[best_fi]))
    print('Best threshold: {0}'.format(best_t))
    
    return (max_setosa, best_fi, best_t)

# We load the data with load_iris from sklearn
data = load_iris()
features = data['data']
feature_names = data['feature_names']
target = data['target']
target_names = data['target_names']
for t,marker,c in zip(range(3),">ox","rgb"):
# We plot each class on its own to get different colored markers
    plt.scatter(features[target == t,0], features[target == t,1], marker=marker, c=c)
plt.show()

# Training with all data
trainModel(features, target)

# Leave one out cross-validation
accuracy = 0.0
for ei in range(len(features)):
# select all but the one at position 'ei':
    training = np.ones(len(features), bool)
    training[ei] = False
    (threshold_setosa, best_fi, best_t) = trainModel(features[training], target[training])
    if features[ei, 2] <= threshold_setosa:
        pred = 'setosa'
    elif features[ei, best_fi] <= best_t:
        pred = 'versicolor'
    else:
        pred = 'virginica'
    if pred == target_names[target[ei]]:
        accuracy += 1
print('Leave-one-out cross-validation accuracy: {0}'.format(accuracy / len(features)))

# 10-fold cross-validation
accuracy = 0.0
rs = ShuffleSplit(n_splits=10, test_size=.10, random_state=0)
for train_index, test_index in rs.split(features):
    (threshold_setosa, best_fi, best_t) = trainModel(features[train_index], target[train_index])
    for idx in test_index:
        if features[idx, 2] <= threshold_setosa:
            pred = 'setosa'
        elif features[idx, best_fi] <= best_t:
            pred = 'versicolor'
        else:
            pred = 'virginica'
        if pred == target_names[target[idx]]:
            accuracy += 1.0 / len(test_index)
print('10-fold cross-validation accuracy: {0}'.format(accuracy / 10))