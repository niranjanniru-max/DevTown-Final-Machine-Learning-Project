import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# 1. Load dataset
data = pd.read_csv(r'D:\ML PROJECTS\customer_retail csv file.csv')
data = data[['Quantity','UnitPrice','Country']]

# 2. Handle missing values
data = data.dropna()

# 3. Encode categorical target
le = LabelEncoder()
data['Country'] = le.fit_transform(data['Country'])

# Features and target
X = data[['Quantity','UnitPrice']]
y = data['Country']

# 4. Visualization: Customer distribution
plt.figure(figsize=(10,6))
data['Country'].value_counts().plot(kind='bar')
plt.title("Customer Distribution by Country")
plt.show()

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Scale numeric features (only once)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

accuracy_scores = {}

# 9. Train and evaluate
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    accuracy_scores[name] = acc
    
    print(f"{name} Accuracy: {acc}")
    print(f"{name} Confusion Matrix:\n{cm}\n")

# 10. Compare performances
plt.bar(accuracy_scores.keys(), accuracy_scores.values())
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

