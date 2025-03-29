# main.py

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import random
import math
import time
from solution import solution
from keras.callbacks import EarlyStopping
import tensorflow as tf
import logging
from sklearn.feature_selection import SelectKBest, f_regression
import seaborn as sns

# Suppress TensorFlow warnings for cleaner output
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Load and preprocess data
data = pd.read_csv('your data path')

# Ensure no missing values
if data.isnull().values.any():
    data = data.dropna()
    print("Dropped rows with missing values.")

# Convert 'Time' to datetime and set as index
data['Time'] = pd.to_datetime(data['Time'])
data.set_index('Time', inplace=True)

# Define feature columns
feature_columns = ['your data columns']

# (Optional) Feature selection can be applied here

# Scale features
scaler = MinMaxScaler()
data[feature_columns] = scaler.fit_transform(data[feature_columns])

# Create sequences for LSTM
def create_sequences(data, feature_columns, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[feature_columns].iloc[i:i + n_steps].values)
        y.append(data['column as target'].iloc[i + n_steps])
    return np.array(X), np.array(y)

n_steps = 60  # Adjust based on memory constraints
X, y = create_sequences(data, feature_columns, n_steps)

# Split data into train, validation, and test sets
train_size = int(len(X) * 0.7)  # 70% for training
val_size = int(len(X) * 0.15)   # 15% for validation
X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]  # Remaining 15% for testing

# Enhanced LSTM model performance function with 6 hyperparameters:
# hyperparameters[0] -> LSTM units (10 to 100)
# hyperparameters[1] -> Dropout rate after LSTM stack (0.0 to 0.5)
# hyperparameters[2] -> Learning rate (1e-5 to 1e-2)
# hyperparameters[3] -> Number of LSTM layers (1 to 3)
# hyperparameters[4] -> Bidirectional flag (if >=0.5, use bidirectional, else unidirectional)
# hyperparameters[5] -> Recurrent dropout (0.0 to 0.5)
def lstm_model_performance(hyperparameters):
    # Extract and adjust hyperparameters:
    lstm_units = int(round(max(10, hyperparameters[0])))
    dropout_rate = np.clip(hyperparameters[1], 0.0, 0.5)
    learning_rate = hyperparameters[2]
    num_layers = int(round(np.clip(hyperparameters[3], 1, 3)))
    bidirectional_flag = hyperparameters[4] >= 0.5  # True if value >= 0.5
    recurrent_dropout = np.clip(hyperparameters[5], 0.0, 0.5)

    model = Sequential()
    model.add(Input(shape=(n_steps, len(feature_columns))))
    
    # Build a stacked LSTM (or bidirectional LSTM) network.
    for i in range(num_layers):
        # For all layers except the last, return sequences.
        return_seq = True if i < num_layers - 1 else False
        lstm_layer = LSTM(lstm_units, activation='relu', return_sequences=return_seq, 
                          recurrent_dropout=recurrent_dropout)
        if bidirectional_flag:
            lstm_layer = Bidirectional(lstm_layer)
        model.add(lstm_layer)
    
    # Add a Dropout layer after the LSTM layers (if desired)
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')

    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    try:
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            verbose=1
        )
        val_loss = model.evaluate(X_val, y_val, verbose=0)
    except Exception as e:
        print(f"Error with hyperparameters {hyperparameters}: {e}")
        val_loss = np.inf
        history = None

    return val_loss, history, model

# Objective function that HHO will minimize
def objective_function(hyperparameters):
    loss, history, _ = lstm_model_performance(hyperparameters)
    return loss

# Harris Hawks Optimization (HHO) function (adapted to the new dimensionality)
def HHO(objf, lb, ub, dim, SearchAgents_no, Max_iter, solution_obj):
    Rabbit_Location = np.zeros(dim)
    Rabbit_Energy = float("inf")
    X = np.random.uniform(0, 1, (SearchAgents_no, dim)) * (ub - lb) + lb
    convergence_curve = np.zeros(Max_iter)

    print(f"HHO is tackling \"{objf.__name__}\"")
    t = 0
    while t < Max_iter:
        for i in range(SearchAgents_no):
            X[i, :] = np.clip(X[i, :], lb, ub)
            solution_obj.fitness_evaluations += 1  # Increment fitness evaluations
            fitness = objf(X[i, :])
            if fitness < Rabbit_Energy:
                Rabbit_Energy = fitness
                Rabbit_Location = X[i, :].copy()
        E1 = 2 * (1 - t / Max_iter)
        for i in range(SearchAgents_no):
            E0 = 2 * random.random() - 1
            Escaping_Energy = E1 * E0
            if abs(Escaping_Energy) >= 1:
                q = random.random()
                rand_Hawk_index = math.floor(SearchAgents_no * random.random())
                X_rand = X[rand_Hawk_index, :]
                if q < 0.5:
                    X[i, :] = X_rand - random.random() * abs(X_rand - 2 * random.random() * X[i, :])
                    solution_obj.mutation_count += 1
                else:
                    X[i, :] = Rabbit_Location - X.mean(0) - random.random() * ((ub - lb) * random.random() + lb)
                    solution_obj.crossover_count += 1
            elif abs(Escaping_Energy) < 1:
                r = random.random()
                if r >= 0.5 and abs(Escaping_Energy) < 0.5:
                    X[i, :] = Rabbit_Location - Escaping_Energy * abs(Rabbit_Location - X[i, :])
                elif r >= 0.5 and abs(Escaping_Energy) >= 0.5:
                    Jump_strength = 2 * (1 - random.random())
                    X1 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X[i, :])
                    solution_obj.fitness_evaluations += 1
                    if objf(X1) < fitness:
                        X[i, :] = X1.copy()
                        solution_obj.crossover_count += 1
        convergence_curve[t] = Rabbit_Energy
        print(f"Iteration {t + 1}/{Max_iter}, Best Fitness: {Rabbit_Energy}")
        t += 1

    solution_obj.best = Rabbit_Energy
    solution_obj.bestIndividual = Rabbit_Location
    solution_obj.convergence = convergence_curve
    return solution_obj

# Set hyperparameter bounds and dimension:
# [lstm_units, dropout_rate, learning_rate, num_layers, bidirectional_flag, recurrent_dropout]
lb = np.array([10, 0.0, 1e-5, 1, 0, 0.0])
ub = np.array([100, 0.5, 1e-2, 3, 1, 0.5])
dim = 6  # Now we have six hyperparameters.
SearchAgents_no = 5    # Adjust the number of search agents if needed.
Max_iter = 5           # Adjust the number of iterations for your application.

# Multiple optimization runs for statistics
num_runs = 3
fitness_values = []
execution_times = []
mutation_counts = []
crossover_counts = []
fitness_evaluations = []

for run in range(num_runs):
    print(f"\nRun {run + 1}/{num_runs}")
    start_time = time.time()
    sol = solution()
    sol = HHO(objective_function, lb, ub, dim, SearchAgents_no, Max_iter, sol)
    fitness_values.append(sol.best)
    execution_times.append(time.time() - start_time)
    mutation_counts.append(sol.mutation_count)
    crossover_counts.append(sol.crossover_count)
    fitness_evaluations.append(sol.fitness_evaluations)

# Compute and print statistics
fitness_values = np.array(fitness_values)
avg_error = np.mean(fitness_values)
best_error = np.min(fitness_values)
worst_error = np.max(fitness_values)
std_error = np.std(fitness_values)
avg_mutation = np.mean(mutation_counts)
avg_crossover = np.mean(crossover_counts)
avg_fitness_eval = np.mean(fitness_evaluations)

print("\nStatistics Across Runs:")
print(f"Average Error: {avg_error}")
print(f"Best Error: {best_error}")
print(f"Worst Error: {worst_error}")
print(f"Standard Deviation: {std_error}")
print(f"Average Mutation Count: {avg_mutation}")
print(f"Average Crossover Count: {avg_crossover}")
print(f"Average Fitness Evaluations: {avg_fitness_eval}")
print(f"Average Execution Time: {np.mean(execution_times):.2f} seconds")

# Evaluate best model on the test set
print("\nTraining the best model found by HHO on the entire training and validation set...")
best_hyperparameters = sol.bestIndividual
_, history, best_model = lstm_model_performance(best_hyperparameters)
y_pred = best_model.predict(X_test)

# Inverse transform to original scale
y_test_scaled = y_test.reshape(-1, 1)
y_pred_scaled = y_pred

# Create dummy features for inverse transforming only 'target'
zeros_other_features = np.zeros((y_test_scaled.shape[0], len(feature_columns) - 1))
y_test_full = np.concatenate((y_test_scaled, zeros_other_features), axis=1)
y_pred_full = np.concatenate((y_pred_scaled, zeros_other_features), axis=1)

y_test_original = scaler.inverse_transform(y_test_full)[:, 0]
y_pred_original = scaler.inverse_transform(y_pred_full)[:, 0]

# Evaluation metrics
mse = mean_squared_error(y_test_original, y_pred_original)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_original, y_pred_original)
epsilon = 1e-10  # Avoid division by zero
mape = np.mean(np.abs((y_test_original - y_pred_original) / (y_test_original + epsilon))) * 100

print(f"\nTest Set Performance:")
print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")

# Plotting results
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(y_test_original, label='Actual target', color='green')
plt.title('Actual target')
plt.xlabel('Time')
plt.ylabel('target')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(y_pred_original, label='Predicted target', color='red')
plt.title('Predicted target')
plt.xlabel('Time')
plt.ylabel('target')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(y_test_original, label='Actual target', color='green')
plt.plot(y_pred_original, label='Predicted target', color='red', alpha=0.7)
plt.title('Actual vs Predicted target')
plt.xlabel('Time')
plt.ylabel('target')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(sol.convergence, label='Convergence Curve', color='purple')
plt.title('HHO Convergence Curve')
plt.xlabel('Iterations')
plt.ylabel('Best Fitness (Validation Loss)')
plt.legend()
plt.grid(True)
plt.show()

runs = range(1, num_runs + 1)
plt.figure(figsize=(10, 6))
plt.bar(runs, mutation_counts, color='skyblue')
plt.title('Number of Mutation Operations per Run')
plt.xlabel('Run')
plt.ylabel('Mutation Count')
plt.xticks(runs)
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(runs, crossover_counts, color='salmon')
plt.title('Number of Crossover Operations per Run')
plt.xlabel('Run')
plt.ylabel('Crossover Count')
plt.xticks(runs)
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(runs, fitness_evaluations, color='lightgreen')
plt.title('Number of Fitness Evaluations per Run')
plt.xlabel('Run')
plt.ylabel('Fitness Evaluations')
plt.xticks(runs)
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(10, 6))
plt.boxplot(fitness_values, patch_artist=True,
            boxprops=dict(facecolor='lightblue'),
            medianprops=dict(color='red'))
plt.title('Distribution of Fitness Values Across Runs')
plt.ylabel('Fitness (Validation Loss)')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(sol.convergence, marker='o', linestyle='-', color='darkblue')
plt.title('Best Fitness per Iteration')
plt.xlabel('Iteration')
plt.ylabel('Fitness (Validation Loss)')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(y_test_original, bins=50, alpha=0.5, label='Actual target', color='green')
plt.hist(y_pred_original, bins=50, alpha=0.5, label='Predicted target', color='red')
plt.title('Histogram of Actual vs Predicted target')
plt.xlabel('target')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

corr_matrix = data[feature_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Features')
plt.show()
