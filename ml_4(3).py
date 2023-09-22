import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OneHotEncoder

data = {
    'Age': ['<=30', '<=30', '31…40', '>40', '>40', '>40', '31…40', '<=30', '<=30', '>40', '<=30', '31…40', '31…40', '>40'],
    'Income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'Student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'Credit_Rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'Buys_Computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'no', 'no']
}

df = pd.DataFrame(data)

df_encoded = pd.get_dummies(df, columns=['Age', 'Income', 'Student', 'Credit_Rating'])

X = df_encoded.drop('Buys_Computer', axis=1)
y = df_encoded['Buys_Computer']

model = DecisionTreeClassifier()
model.fit(X, y)

tree_depth = model.get_depth()
print("Tree Depth:", tree_depth)

feature_names = X.columns.tolist()

plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, feature_names=feature_names, class_names=['No', 'Yes'])
plt.show()
