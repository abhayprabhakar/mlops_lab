import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
import joblib

data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

joblib.dump(model, 'diabetes_model.joblib')

loaded_model = joblib.load('diabetes_model.joblib')
loaded_y_pred = loaded_model.predict(X_test)
loaded_mse = mean_squared_error(y_test, loaded_y_pred)
print(f"Loaded Model MSE: {loaded_mse}")