# LOADING THE DATASET
import pandas as pd 
import numpy as np 
df=pd.read_csv("Adobe_Data.csv", header=0)
naames=["Date","Adj_Close","Close","High","Low","Open","Volume"]
# checking the 5 row 
print(df.head())
# checking the non null value 
print(df.info())
# checking the columns name 
print(df.columns)
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
# reset the index of the date
df=df.sort_values("Date").reset_index(drop=True)
print(df["Date"].head())
print(df["Date"].dtype)
#checking the shape
print("shape of the data:", df.shape)
# data range 
print("data range:", df["Date"].min(), "to", df["Date"].max())
# checking the missing value 
print("missing values:\n",df.isnull().sum())
# checking the statistical summary
print("statistical summary:",df.describe())
#---------------Feature Engineering------------------
df_fetured=df.copy()
#  capture market movement 
df_fetured["log_return"]=np.log(df_fetured["Adj_Close"] / df_fetured["Adj_Close"].shift(1))
# Daily flucatuation(volatility)
df_fetured["h1_range"]=(df_fetured["High"] - df_fetured["Low"])/df_fetured["Close"]
# moving average (5 days)
df_fetured["ma_5"] = df_fetured["Close"].rolling(window=5).mean()
# moving average (10 days )
df_fetured["ma_10"] = df_fetured["Close"].rolling(window=10).mean()
# moving average(20 days)
df_fetured["ma_20"]=df_fetured["Close"].rolling(window=20).mean()
# momentum indicator
# price difference
delta=df_fetured["Close"].diff()
# gain and loss 
gain=np.where(delta>0,delta,0)
loss=np.where(delta<0,-delta,0)
# rolling average(14 days)
roll_up=pd.Series(gain).rolling(window=14).mean()
roll_down=pd.Series(loss).rolling(window=14).mean()
# rsi calculation mean how much body move up vs down
rs=roll_up/(roll_down +1e-9)
df_fetured["rsi_14"]=100.0 - (100.0 / (1.0 + rs))
# drop the nana values
df_fetured=df_fetured.dropna().reset_index(drop=True)
print(df_fetured.head())
print("shape of the feature engineered data:", df_fetured.shape)
print("missing values in feature engineered data:\n",df_fetured.isnull().sum())
#choosing the input + target
features=["Open","High","Low","Close","Volume","log_return","h1_range","ma_5","ma_10","ma_20","rsi_14"]
target_col="Adj_Close"  
data =df_fetured[["Date"]+features+[target_col]].copy()
# walk forward split
train_end_date="2021-12-31"
val_end_date="2022-12-31"
# then traing data 
train_df=data[data["Date"]<=train_end_date].copy()
val_df=data[(data["Date"]>train_end_date) & (data["Date"]<=val_end_date)].copy()
# test data 
test_df=data[data["Date"]>val_end_date].copy()
print(train_df.shape, val_df.shape, test_df.shape)
print("train date range:", train_df["Date"].min(), "to", train_df["Date"].max())
print("validation date range:", val_df["Date"].min(), "to", val_df["Date"].max())
print("test date range:", test_df["Date"].min(), "to", test_df["Date"].max())
print(train_df['Date'].min(), train_df['Date'].max())
print(test_df['Date'].min(), test_df['Date'].max())
# scaaling 
from sklearn.preprocessing import MinMaxScaler
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
x_train_raw = x_scaler.fit_transform(train_df[features].values)
y_train_raw = y_scaler.fit_transform(train_df[[target_col]].values)
x_val_raw = x_scaler.transform(val_df[features].values)
y_val_raw = y_scaler.transform(val_df[[target_col]].values)
x_test_raw = x_scaler.transform(test_df[features].values)
y_test_raw = y_scaler.transform(test_df[[target_col]].values)
X_train = x_train_raw
X_val   = x_val_raw
X_test  = x_test_raw

y_train = y_train_raw
y_val   = y_val_raw
y_test  = y_test_raw
print("x_train shape:", x_train_raw.shape)
print("y_train shape:", y_train_raw.shape)
print("x_val shape:", x_val_raw.shape)
print("y_val shape:", y_val_raw.shape)
print("x_test shape:", x_test_raw.shape)
print("y_test shape:", y_test_raw.shape)
# sliding window 
def create_sliding_window_data(X, y, window_size):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:(i + window_size)])
        ys.append(y[i + window_size])
    return np.array(Xs), np.array(ys)
window_size = 60
X_train_sw, y_train_sw = create_sliding_window_data(X_train, y_train, window_size)
X_val_sw, y_val_sw = create_sliding_window_data(X_val, y_val, window_size)
X_test_sw, y_test_sw = create_sliding_window_data(X_test, y_test, window_size)
print("X_train_sw shape:", X_train_sw.shape)
print("y_train_sw shape:", y_train_sw.shape)
print("X_val_sw shape:", X_val_sw.shape)
print("y_val_sw shape:", y_val_sw.shape)
print("X_test_sw shape:", X_test_sw.shape)
print("y_test_sw shape:", y_test_sw.shape)


# SECTION 1: MODEL ARCHITECTURES (Vanilla RNN, LSTM, GRU)

import tensorflow as tf  # type: ignore
from tensorflow import keras  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, LSTM, GRU  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def build_vanilla_rnn(input_shape, hidden_units=64, dropout_rate=0.2, learning_rate=1e-3):
    
    model = Sequential([
        SimpleRNN(hidden_units, activation='tanh', return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        SimpleRNN(hidden_units//2, activation='tanh'),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer for regression
    ], name='Vanilla_RNN')
    
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='mse',
                  metrics=['mae', 'mape'])
    return model

def build_lstm_model(input_shape, hidden_units=64, dropout_rate=0.2, learning_rate=1e-3, stacked=True):
    
    
    model = Sequential(name='LSTM_Model')
    
    if stacked:
        # Two stacked LSTM layers
        model.add(LSTM(hidden_units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(hidden_units//2, return_sequences=False))
        model.add(Dropout(dropout_rate))
    else:
        # Single LSTM layer
        model.add(LSTM(hidden_units, return_sequences=False, input_shape=input_shape))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='mse',
                  metrics=['mae', 'mape'])
    return model

def build_gru_model(input_shape, hidden_units=64, dropout_rate=0.2, learning_rate=1e-3, stacked=True):
    model = Sequential(name='GRU_Model')
    
    if stacked:
        # Two stacked GRU layers
        model.add(GRU(hidden_units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        model.add(GRU(hidden_units//2, return_sequences=False))
        model.add(Dropout(dropout_rate))
    else:
        # Single GRU layer
        model.add(GRU(hidden_units, return_sequences=False, input_shape=input_shape))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='mse',
                  metrics=['mae', 'mape'])
    return model

print("\n" + "="*80)
print("MODEL ARCHITECTURES DEFINED")
print("="*80)


# SECTION 2: BASELINE MODELS (Naive, ARIMA, Simple MLP)


def naive_baseline(y_train, y_val, y_test):
    
    # For sliding window data, the last value in sequence predicts next
    y_val_pred_naive = y_val_raw[window_size-1:-1]  # Shift by 1
    y_test_pred_naive = y_test_raw[window_size-1:-1]
    
    # Adjust to match lengths
    y_val_pred_naive = np.concatenate([y_val_raw[window_size-1:window_size], y_val_raw[window_size:-1]])
    y_test_pred_naive = np.concatenate([y_test_raw[window_size-1:window_size], y_test_raw[window_size:-1]])
    
    return y_val_pred_naive[:len(y_val_sw)], y_test_pred_naive[:len(y_test_sw)]

def build_simple_mlp(input_shape, hidden_units=64, dropout_rate=0.2, learning_rate=1e-3):
    
    model = Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        Dense(hidden_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        Dropout(dropout_rate),
        Dense(1)
    ], name='MLP_Baseline')
    
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='mse',
                  metrics=['mae', 'mape'])
    return model

print("\n" + "="*80)
print("BASELINE MODELS DEFINED")
print("="*80)


# SECTION 3: EVALUATION METRICS

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true, y_pred, model_name="Model"):
    
    # Inverse transform to original scale for interpretable metrics
    y_true_orig = y_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_orig = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true_orig - y_pred_orig) / (y_true_orig + 1e-9))) * 100
    
    # sMAPE (Symmetric MAPE)
    smape = np.mean(2.0 * np.abs(y_pred_orig - y_true_orig) / (np.abs(y_true_orig) + np.abs(y_pred_orig) + 1e-9)) * 100
    
    # R-squared
    r2 = r2_score(y_true_orig, y_pred_orig)
    
    # Directional Accuracy (up/down prediction)
    y_true_diff = np.diff(y_true_orig)
    y_pred_diff = np.diff(y_pred_orig)
    directional_accuracy = np.mean((y_true_diff * y_pred_diff) > 0) * 100
    
    metrics = {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'sMAPE': smape,
        'R2': r2,
        'Directional_Accuracy': directional_accuracy
    }
    
    return metrics

def calculate_economic_metrics(y_true, y_pred, initial_capital=10000):
    
    y_true_orig = y_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_orig = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    # Calculate returns
    true_returns = np.diff(y_true_orig) / y_true_orig[:-1]
    pred_direction = np.diff(y_pred_orig)
    
    # Trading strategy: long if predicted up, flat otherwise
    strategy_returns = np.where(pred_direction > 0, true_returns, 0)
    
    # Cumulative PnL
    cumulative_returns = np.cumprod(1 + strategy_returns)
    final_capital = initial_capital * cumulative_returns[-1]
    total_return = (final_capital - initial_capital) / initial_capital * 100
    
    # Max Drawdown
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = np.min(drawdown) * 100
    
    return {
        'Total_Return_%': total_return,
        'Final_Capital': final_capital,
        'Max_Drawdown_%': max_drawdown
    }

print("\n" + "="*80)
print("EVALUATION METRICS DEFINED")
print("="*80)


# SECTION 4: TRAINING PIPELINE


def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, patience=10, verbose=1):
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=verbose)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=verbose)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=verbose
    )
    
    return model, history

print("\n" + "="*80)
print("TRAINING PIPELINE DEFINED")
print("="*80)


# SECTION 5: VISUALIZATION FUNCTIONS

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

def plot_training_history(history, model_name):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title(f'{model_name} - Training History (Loss)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].plot(history.history['mae'], label='Train MAE', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title(f'{model_name} - Training History (MAE)', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {model_name}_training_history.png")

def plot_predictions(y_true, y_pred, model_name, dataset_name='Test', num_points=200):
    
    y_true_orig = y_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_orig = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    # Plot last num_points for clarity
    y_true_plot = y_true_orig[-num_points:]
    y_pred_plot = y_pred_orig[-num_points:]
    
    plt.figure(figsize=(14, 6))
    plt.plot(y_true_plot, label='Actual Price', linewidth=2, alpha=0.8)
    plt.plot(y_pred_plot, label='Predicted Price', linewidth=2, alpha=0.8, linestyle='--')
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Stock Price (USD)', fontsize=12)
    plt.title(f'{model_name} - Actual vs Predicted ({dataset_name} Set, Last {num_points} Points)', 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{model_name}_predictions_{dataset_name}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {model_name}_predictions_{dataset_name}.png")

def plot_prediction_error(y_true, y_pred, model_name, dataset_name='Test'):
    
    y_true_orig = y_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_orig = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    errors = y_true_orig - y_pred_orig
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Error over time
    axes[0].plot(errors, linewidth=1, alpha=0.7)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Time Steps', fontsize=12)
    axes[0].set_ylabel('Prediction Error (USD)', fontsize=12)
    axes[0].set_title(f'{model_name} - Prediction Errors Over Time', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Error distribution
    axes[1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Prediction Error (USD)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'{model_name} - Error Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_errors_{dataset_name}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {model_name}_errors_{dataset_name}.png")

print("\n" + "="*80)
print("VISUALIZATION FUNCTIONS DEFINED")
print("="*80)


# SECTION 6: EXPERIMENT WITH DIFFERENT CONFIGURATIONS


# Store all results
all_results = []
all_histories = {}
all_predictions = {}

# Configuration for experiments
input_shape = (X_train_sw.shape[1], X_train_sw.shape[2])  # (window_size, num_features)

# Hyperparameter configurations to test
configs = [
    {'name': 'RNN_64', 'model_type': 'RNN', 'hidden_units': 64, 'dropout': 0.2, 'lr': 1e-3},
    {'name': 'LSTM_64', 'model_type': 'LSTM', 'hidden_units': 64, 'dropout': 0.2, 'lr': 1e-3},
    {'name': 'LSTM_128', 'model_type': 'LSTM', 'hidden_units': 128, 'dropout': 0.3, 'lr': 5e-4},
    {'name': 'GRU_64', 'model_type': 'GRU', 'hidden_units': 64, 'dropout': 0.2, 'lr': 1e-3},
    {'name': 'GRU_128', 'model_type': 'GRU', 'hidden_units': 128, 'dropout': 0.3, 'lr': 5e-4},
    {'name': 'MLP_Baseline', 'model_type': 'MLP', 'hidden_units': 64, 'dropout': 0.2, 'lr': 1e-3}
]

print("\n" + "="*80)
print("STARTING MODEL TRAINING AND EVALUATION")
print("="*80)

# SECTION 7: TRAIN AND EVALUATE ALL MODELS


for config in configs:
    print(f"\n{'='*80}")
    print(f"Training: {config['name']}")
    print(f"{'='*80}")
    
    # Build model based on type
    if config['model_type'] == 'RNN':
        model = build_vanilla_rnn(input_shape, config['hidden_units'], config['dropout'], config['lr'])
    elif config['model_type'] == 'LSTM':
        model = build_lstm_model(input_shape, config['hidden_units'], config['dropout'], config['lr'], stacked=True)
    elif config['model_type'] == 'GRU':
        model = build_gru_model(input_shape, config['hidden_units'], config['dropout'], config['lr'], stacked=True)
    elif config['model_type'] == 'MLP':
        model = build_simple_mlp(input_shape, config['hidden_units'], config['dropout'], config['lr'])
    
    # Display model architecture
    print(f"\nModel Architecture:")
    model.summary()
    
    # Train model
    print(f"\nTraining {config['name']}...")
    model, history = train_model(
        model, X_train_sw, y_train_sw, X_val_sw, y_val_sw,
        epochs=100, batch_size=32, patience=15, verbose=1
    )
    
    # Store history
    all_histories[config['name']] = history
    
    # Make predictions
    y_val_pred = model.predict(X_val_sw, verbose=0).flatten()
    y_test_pred = model.predict(X_test_sw, verbose=0).flatten()
    
    # Store predictions
    all_predictions[config['name']] = {
        'val': y_val_pred,
        'test': y_test_pred
    }
    
    # Calculate metrics for validation set
    val_metrics = calculate_metrics(y_val_sw.flatten(), y_val_pred, f"{config['name']}_Val")
    
    # Calculate metrics for test set
    test_metrics = calculate_metrics(y_test_sw.flatten(), y_test_pred, f"{config['name']}_Test")
    
    # Calculate economic metrics
    economic_metrics = calculate_economic_metrics(y_test_sw.flatten(), y_test_pred)
    
    # Combine all metrics
    combined_metrics = {**test_metrics, **economic_metrics}
    all_results.append(combined_metrics)
    
    # Print results
    print(f"\n{config['name']} - Validation Metrics:")
    print(f"  RMSE: ${val_metrics['RMSE']:.2f}")
    print(f"  MAE: ${val_metrics['MAE']:.2f}")
    print(f"  MAPE: {val_metrics['MAPE']:.2f}%")
    print(f"  R²: {val_metrics['R2']:.4f}")
    print(f"  Directional Accuracy: {val_metrics['Directional_Accuracy']:.2f}%")
    
    print(f"\n{config['name']} - Test Metrics:")
    print(f"  RMSE: ${test_metrics['RMSE']:.2f}")
    print(f"  MAE: ${test_metrics['MAE']:.2f}")
    print(f"  MAPE: {test_metrics['MAPE']:.2f}%")
    print(f"  R²: {test_metrics['R2']:.4f}")
    print(f"  Directional Accuracy: {test_metrics['Directional_Accuracy']:.2f}%")
    print(f"  Total Return: {economic_metrics['Total_Return_%']:.2f}%")
    print(f"  Max Drawdown: {economic_metrics['Max_Drawdown_%']:.2f}%")
    
    # Visualizations
    print(f"\nGenerating visualizations for {config['name']}...")
    plot_training_history(history, config['name'])
    plot_predictions(y_test_sw.flatten(), y_test_pred, config['name'], 'Test', num_points=200)
    plot_prediction_error(y_test_sw.flatten(), y_test_pred, config['name'], 'Test')
    
    print(f"\n{config['name']} completed!")


# SECTION 8: NAIVE BASELINE EVALUATION

print(f"\n{'='*80}")
print("Evaluating NAIVE BASELINE")
print(f"{'='*80}")

# Naive predictions (shift by 1)
y_val_pred_naive, y_test_pred_naive = naive_baseline(y_train_sw, y_val_sw, y_test_sw)

# Calculate metrics for naive baseline
naive_test_metrics = calculate_metrics(y_test_sw.flatten(), y_test_pred_naive, "Naive_Baseline")
naive_economic = calculate_economic_metrics(y_test_sw.flatten(), y_test_pred_naive)
naive_combined = {**naive_test_metrics, **naive_economic}
all_results.append(naive_combined)

print(f"\nNaive Baseline - Test Metrics:")
print(f"  RMSE: ${naive_test_metrics['RMSE']:.2f}")
print(f"  MAE: ${naive_test_metrics['MAE']:.2f}")
print(f"  MAPE: {naive_test_metrics['MAPE']:.2f}%")
print(f"  R²: {naive_test_metrics['R2']:.4f}")
print(f"  Directional Accuracy: {naive_test_metrics['Directional_Accuracy']:.2f}%")


# SECTION 9: RESULTS COMPARISON AND SUMMARY
print(f"\n{'='*80}")
print("COMPREHENSIVE RESULTS COMPARISON")
print(f"{'='*80}")

# Create results DataFrame
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('RMSE')

print("\n" + "="*80)
print("ALL MODELS - TEST SET PERFORMANCE (Sorted by RMSE)")
print("="*80)
print(results_df.to_string(index=False))

# Save results to CSV
results_df.to_csv('adobe_model_comparison_results.csv', index=False)
print("\n Results saved to: adobe_model_comparison_results.csv")


# SECTION 10: COMPARATIVE VISUALIZATIONS


# Plot 1: Model comparison - RMSE and MAE
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

models = results_df['Model'].values
rmse_values = results_df['RMSE'].values
mae_values = results_df['MAE'].values

axes[0].barh(models, rmse_values, color='steelblue', alpha=0.8)
axes[0].set_xlabel('RMSE (USD)', fontsize=12)
axes[0].set_title('Model Comparison - RMSE', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')

axes[1].barh(models, mae_values, color='coral', alpha=0.8)
axes[1].set_xlabel('MAE (USD)', fontsize=12)
axes[1].set_title('Model Comparison - MAE', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('model_comparison_errors.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: model_comparison_errors.png")

# Plot 2: R² and Directional Accuracy comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

r2_values = results_df['R2'].values
dir_acc_values = results_df['Directional_Accuracy'].values

axes[0].barh(models, r2_values, color='green', alpha=0.8)
axes[0].set_xlabel('R² Score', fontsize=12)
axes[0].set_title('Model Comparison - R² Score', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')

axes[1].barh(models, dir_acc_values, color='purple', alpha=0.8)
axes[1].set_xlabel('Directional Accuracy (%)', fontsize=12)
axes[1].set_title('Model Comparison - Directional Accuracy', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('model_comparison_accuracy.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Saved: model_comparison_accuracy.png")

# Plot 3: Economic metrics comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

return_values = results_df['Total_Return_%'].values
drawdown_values = results_df['Max_Drawdown_%'].values

axes[0].barh(models, return_values, color='gold', alpha=0.8)
axes[0].set_xlabel('Total Return (%)', fontsize=12)
axes[0].set_title('Model Comparison - Total Return', fontsize=14, fontweight='bold')
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=1)
axes[0].grid(True, alpha=0.3, axis='x')

axes[1].barh(models, drawdown_values, color='crimson', alpha=0.8)
axes[1].set_xlabel('Max Drawdown (%)', fontsize=12)
axes[1].set_title('Model Comparison - Max Drawdown', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('model_comparison_economic.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Saved: model_comparison_economic.png")

# BEST MODEL SELECTION AND FINAL ANALYSIS
best_model_name = results_df.iloc[0]['Model']
best_rmse = results_df.iloc[0]['RMSE']
best_r2 = results_df.iloc[0]['R2']

print(f"\n{'='*80}")
print("BEST MODEL SELECTION")
print(f"{'='*80}")
print(f" Best Model: {best_model_name}")
print(f"   RMSE: ${best_rmse:.2f}")
print(f"   R²: {best_r2:.4f}")
print(f"   Directional Accuracy: {results_df.iloc[0]['Directional_Accuracy']:.2f}%")
print(f"   Total Return: {results_df.iloc[0]['Total_Return_%']:.2f}%")



print(f"\n{'='*80}")
print("REGIME ANALYSIS - Performance by Time Period")
print(f"{'='*80}")

# Split test data by years for regime analysis
test_dates = test_df['Date'].iloc[window_size:window_size+len(y_test_sw)]

# Define regimes
regimes = {
    '2023': (test_dates >= '2023-01-01') & (test_dates < '2024-01-01'),
    '2024+': test_dates >= '2024-01-01'
}

# Analyze best model across regimes
best_model_key = best_model_name.replace('_Test', '').replace('_Val', '')
if best_model_key in all_predictions:
    y_test_pred_best = all_predictions[best_model_key]['test']
    
    print(f"\nRegime-wise performance for {best_model_key}:")
    for regime_name, regime_mask in regimes.items():
        if regime_mask.sum() > 0:
            regime_true = y_test_sw.flatten()[regime_mask.values[:len(y_test_sw)]]
            regime_pred = y_test_pred_best[regime_mask.values[:len(y_test_sw)]]
            
            if len(regime_true) > 10:  
                regime_metrics = calculate_metrics(regime_true, regime_pred, f"{best_model_key}_{regime_name}")
                print(f"\n  {regime_name}:")
                print(f"    RMSE: ${regime_metrics['RMSE']:.2f}")
                print(f"    MAE: ${regime_metrics['MAE']:.2f}")
                print(f"    R²: {regime_metrics['R2']:.4f}")
                print(f"    Directional Accuracy: {regime_metrics['Directional_Accuracy']:.2f}%")





print(f"\n{'='*80}")
print("ANALYSIS COMPLETE - ADOBE STOCK PRICE FORECASTING")
print(f"{'='*80}")
print(f"\n Models Evaluated: {len(configs) + 1} (including Naive baseline)")
print(f" Best Performing Model: {best_model_name}")
print(f" Results saved to: adobe_model_comparison_results.csv")
print(f" Generated {len(configs) * 3 + 3} visualization plots")
print(f"\n{'='*80}")
