import pandas as pd
from sklearn.tree import DecisionTreeClassifier
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
