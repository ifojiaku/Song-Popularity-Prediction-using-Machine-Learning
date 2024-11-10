import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the data from the CSV file
file_paths = [r"C:\Users\navee\Downloads\ML Datasets - Copy of top - 10s OG.csv", r"C:\Users\navee\Downloads\ML Datasets - Best Songs on Spotify from 2000-2023 - adjusted.csv"]  # List of file paths
data1 = [pd.read_csv(file) for file in file_paths]
data = pd.concat(data1, ignore_index=True)
data.to_csv('combined_data.csv', index=False)


# Step 2: Preprocess the data
# Define thresholds
high_threshold = 85
low_threshold = 65

# Assign categories based on thresholds
def categorize(value):
    if value >= high_threshold:
        return 1
    elif value >= low_threshold:
        return 2
    else:
        return 3

data['pop_adjusted'] = data['pop'].apply(categorize)

# Write back to CSV
data.to_csv('your_file_with_categories.csv', index=False)

#data = data = pd.read_csv(r"C:\Users\navee\Downloads\ML Datasets - Copy of top - 10s OG.csv")

# Write back to CSV

# Step 2: Preprocess Data
X = data.drop(['id','title','top genre','year','artist','pop_adjusted','pop'], axis=1)  # Features
y = data['pop_adjusted']               # Target variable

# Step 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Define the parameter grid
param_grid = {
    'max_depth': [1,2,3],
    'min_samples_split': [1,2,3],
    'min_samples_leaf': [1,2,3]
}

# Step 5: Create the grid search model
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='accuracy')

# Step 6: Train the grid search model
grid_search.fit(X_train, y_train)

# Step 7: Get the best parameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Step 8: Evaluate the best model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Best Parameters:", best_params)
print("Accuracy:", accuracy)

# Step 9: Visualize the decision tree
plt.figure(figsize=(20,10))
plot_tree(best_model, filled=True, feature_names=X.columns, class_names=[str(c) for c in best_model.classes_])
plt.show()

# Step 6: Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)

# Step 7: Calculate R^2 Score
r2 = r2_score(y_test, y_pred)
print("R^2 Score:", r2)
