import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("diabetes.csv")

print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset shape:", df.shape)

X = df.drop("Outcome", axis = 1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\nModel accuracy:", accuracy)

joblib.dump(model, "diabetes.pkl")

print("\nModel successfully saved as diabetes.pkl")