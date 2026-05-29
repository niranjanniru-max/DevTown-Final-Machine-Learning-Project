import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


data = pd.read_csv(r'D:\ML PROJECTS\customer_retail csv file.csv')
data = data[['Quantity','UnitPrice','Country']]


data = data.dropna()


le = LabelEncoder()
data['Country'] = le.fit_transform(data['Country'])


X = data[['Quantity','UnitPrice']]
y = data['Country']

plt.figure(figsize=(10,6))
data['Country'].value_counts().plot(kind='bar')
plt.title("Customer Distribution by Country")
plt.show()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

accuracy_scores = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    accuracy_scores[name] = acc

    print(f"{name} Accuracy: {acc}")
    print(f"{name} Confusion Matrix:\n{cm}\n")


plt.bar(accuracy_scores.keys(), accuracy_scores.values())
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

#Output :
'''
Logistic Regression Accuracy: 0.9142846598143604
Logistic Regression Confusion Matrix:
[[    0     0     0 ...     0   252     0]
 [    0     0     0 ...     0    80     0]
 [    0     0     0 ...     0     4     0]
 ...
 [    0     0     0 ...     0    14     0]
 [    0     0     0 ...     0 99092     0]
 [    0     0     0 ...     0    89     0]]

Decision Tree Accuracy: 0.9143861526821797
Decision Tree Confusion Matrix:
[[    8     0     0 ...     0   227     0]
 [    0     0     0 ...     0    78     0]
 [    0     0     0 ...     0     4     0]
 ...
 [    0     0     0 ...     0    14     0]
 [   34     0     0 ...     0 98890     0]
 [    0     0     0 ...     0    89     0]]

KNN Accuracy: 0.908001328633906
KNN Confusion Matrix:
[[    6     0     0 ...     0   216     0]
 [    0     2     0 ...     0    73     0]
 [    0     0     0 ...     0     4     0]
 ...
 [    0     0     0 ...     0    14     0]
 [   16     5     0 ...     0 98153     0]
 [    0     0     0 ...     0    89     0]]
'''
