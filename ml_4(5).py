import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

data_df = pd.read_excel("embeddingsdata.xlsx")

X_features = data_df[['embed_0', 'embed_1']]
y_target = data_df['Label']

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.3, random_state=42)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"Training Set Accuracy: {train_acc}")
print(f"Test Set Accuracy: {test_acc}")

class_labels = data_df['Label'].unique().astype(str).tolist()

plt.figure(figsize=(10, 6))
plot_tree(model, filled=True, feature_names=['embed_0', 'embed_1'], class_names=class_labels)
plt.title("Decision Tree")
plt.show()
