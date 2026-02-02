# train_model.py
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# 1. Create deterministic training data
# Features: [x1, x2, x3]
X = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [5, 6, 7]
])

# Target variable
y = np.array([10, 14, 18, 22, 26])

# 2. Train model
model = LinearRegression()
model.fit(X, y)

# 3. Save model artifact
joblib.dump(model, "model.pkl")

print("Model trained and saved as model.pkl")