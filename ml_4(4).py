import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

data = pd.read_excel("embeddingsdata.xlsx")

features = data[['embed_0', 'embed_1']]
target = data['Label']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

tree_model = DecisionTreeClassifier()

tree_model.fit(X_train, y_train)

train_accuracy = tree_model.score(X_train, y_train)
test_accuracy = tree_model.score(X_test, y_test)

print(f"Training Set Accuracy: {train_accuracy}")
print(f"Test Set Accuracy: {test_accuracy}")

class_names = data['Label'].unique().astype(str).tolist()

plt.figure(figsize=(10, 6))
plot_tree(tree_model, filled=True, feature_names=['embed_0', 'embed_1'], class_names=class_names)
plt.title("Decision Tree")
plt.show()
