import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    "Age": [25, 45, 30, 35, 50],
    "Balance": [2000, 5000, 3000, 4000, 1000],
    "Churn": [0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df[["Age", "Balance"]]
y = df["Churn"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy)
