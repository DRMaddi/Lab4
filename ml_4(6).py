import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

data = pd.read_excel("embeddingsdata.xlsx")

model_entropy = DecisionTreeClassifier(criterion="entropy")

binary_data = data[data['Label'].isin([0, 1])]
X_features = binary_data[['embed_1', 'embed_2']]
y_target = binary_data['Label']

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.3, random_state=42)
model_entropy.fit(X_train, y_train)

train_acc = model_entropy.score(X_train, y_train)
test_acc = model_entropy.score(X_test, y_test)

print(f"Training Set Accuracy (criterion='entropy'): {train_acc}")
print(f"Test Set Accuracy (criterion='entropy'): {test_acc}")

class_labels = ['No', 'Yes']

plt.figure(figsize=(20, 10))
plot_tree(model_entropy, filled=True, feature_names=['embed_1', 'embed_2'], class_names=class_labels, rounded=True)
plt.show()
