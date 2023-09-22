import pandas as pd
import numpy as np
from math import log2

data = {
    'Age': ['<=30', '<=30', '31…40', '>40', '>40', '>40', '31…40', '<=30', '<=30', '>40', '<=30', '31…40', '31…40', '>40'],
    'Income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'Student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'Credit_Rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'Buys_Computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'no', 'no']
}

df = pd.DataFrame(data)

def calculate_entropy(data):
    total_samples = len(data)
    value_counts = data.value_counts()
    entropy = 0
    
    for count in value_counts:
        probability = count / total_samples
        entropy += -probability * log2(probability)
    
    return entropy

entropy_target = calculate_entropy(df['Buys_Computer'])

def calculate_information_gain(data, feature_name, target_name):
    unique_values = data[feature_name].unique()
    total_entropy = calculate_entropy(data[target_name])
    weighted_entropy = 0
    
    for value in unique_values:
        subset = data[data[feature_name] == value]
        probability = len(subset) / len(data)
        weighted_entropy += probability * calculate_entropy(subset[target_name])
    
    information_gain = total_entropy - weighted_entropy
    return information_gain

features = ['Age', 'Income', 'Student', 'Credit_Rating']
information_gains = {}

for feature in features:
    information_gains[feature] = calculate_information_gain(df, feature, 'Buys_Computer')

root_node = max(information_gains, key=information_gains.get)

print(information_gains)
print("The first feature to select for the root node is:", root_node)
