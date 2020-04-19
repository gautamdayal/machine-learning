import pandas as pd
import numpy as np

data = pd.DataFrame({
    'outlook':['sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy', 'overcast', 'sunny', 'sunny', 'rainy', 'sunny', 'overcast', 'overcast', 'rainy'],
    'temperature':['hot', 'hot', 'hot', 'medium', 'cool', 'cool', 'cool', 'medium', 'cool', 'medium', 'medium', 'medium', 'hot', 'medium'],
    'humidity':['high', 'high', 'high', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'high'],
    'wind':['weak', 'strong', 'weak', 'weak', 'weak', 'strong', 'strong', 'weak', 'weak', 'weak', 'strong', 'strong', 'weak', 'strong'],
    'play':[False, False, True, True, True, False, True, False, True, True, True, True, True, False]})

def getEntropy(feature):
    total, counts = np.unique(feature,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(total))])
    return(entropy)

def getInformationGain(data, feature):
    dataset_entropy = getEntropy(data['play'])
    values,counts= np.unique(data[feature],return_counts=True)
    weighted_entropy = np.sum([(counts[i]/np.sum(counts))*getEntropy(data.where(data[feature]==values[i]).dropna()['play']) for i in range(len(values))])
    information_gain = dataset_entropy - weighted_entropy
    return information_gain

def IterativeDichotomiser3(data, dataset, features, parent_node = None):
    if len(np.unique(data['play'])) <= 1:
        return np.unique(data['play'])[0]

    elif len(data)==0:
        return np.unique(dataset['play'])[np.argmax(np.unique(dataset['play'],return_counts=True)[1])]

    elif len(features) ==0:
        return parent_node

    else:
        parent_node = np.unique(data['play'])[np.argmax(np.unique(data['play'],return_counts=True)[1])]

        info_gain_values = [getInformationGain(data,feature) for feature in features]
        best_feature_i = np.argmax(info_gain_values)
        best_feature = features[best_feature_i]

        tree = {best_feature:{}}
        features = [i for i in features if i != best_feature]

        for value in np.unique(data[best_feature]):
            value = value

            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = IterativeDichotomiser3(sub_data, dataset, features, parent_node)
            tree[best_feature][value] = subtree
        return(tree)

print(IterativeDichotomiser3(data, data, data[['outlook', 'temperature', 'humidity', 'wind']]))
