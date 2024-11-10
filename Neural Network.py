import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
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
data = data.iloc[:, :-2]

# Assume the last column is the target ordinal variable
X = data.iloc[:, 5:-1].astype(int)  # Input features
y = data.iloc[:, -1]  # Target labels

# Convert target labels to integers
y_unique = sorted(y.unique())
y = y.apply(lambda x: y_unique.index(x)).astype(int)

# Convert DataFrame to numpy arrays
X = X.values
y = y.values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X.astype(int), y.astype(int), test_size=0.2, random_state=42)

# Step 3: Define custom loss function for ordinal classification

def ordinal_cross_entropy(y_true, y_pred):
    #print(y_true.dtype)
    y_true = tf.cast(y_true, tf.int32)
    # Convert predicted probabilities to cumulative probabilities
    cum_probs = tf.cumsum(y_pred, axis=-1)
    
    # Convert true labels to one-hot encoding
    y_true_one_hot = tf.one_hot(y_true, depth=y_pred.shape[1])
    
    # Compute the cross-entropy loss
    loss = -tf.reduce_sum(y_true_one_hot * tf.math.log(cum_probs + 1e-9), axis=-1)
    
    # Sum the losses for all samples and divide by the number of samples
    return tf.reduce_mean(loss)
def ordinal_softmax(x):
    # Compute softmax along the last axis
    softmax_output = K.softmax(x, axis=-1)
    
    # Compute cumulative sum of softmax probabilities
    cumulative_probs = K.cumsum(softmax_output, axis=-1)
    
    # Shift cumulative probabilities by 0.5 and clip to ensure they are between 0 and 1
    shifted_cum_probs = K.clip(cumulative_probs - 0.5, 0, 1)
    
    return shifted_cum_probs
# Step 4: Define the neural network architecture
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    #Dense(32, activation='relu'),
    Dense(len(y_unique), activation=ordinal_softmax)  # Output layer with softmax activation
])

# Step 5: Compile the model with custom loss function
model.compile(optimizer='adam', loss=ordinal_cross_entropy, metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))