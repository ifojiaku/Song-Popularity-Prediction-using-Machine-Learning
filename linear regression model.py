import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score

# Load your CSV data
file_paths = [r"C:\Users\navee\Downloads\ML Datasets - Copy of top - 10s OG.csv", r"C:\Users\navee\Downloads\ML Datasets - Best Songs on Spotify from 2000-2023 - adjusted.csv"]  # List of file paths
data1 = [pd.read_csv(file) for file in file_paths]
data = pd.concat(data1, ignore_index=True)
data.to_csv('combined_data.csv', index=False)

# Assuming your target column is 'target' and the rest are features
X = data.drop(['id','title','top genre','year','artist','pop'], axis=1)  # Features
y = data['pop']

# Step 1: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Define the model
model = Ridge()

# Step 3: Hyperparameter Optimization using GridSearchCV
param_grid = {
    'alpha': [0.1, 1.0, 10.0],  # Regularization strength
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)

# Best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the model with best hyperparameters
best_model = Ridge(**best_params)
best_model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

r2 = r2_score(y_test, y_pred)
print("R^2 Score:", r2)