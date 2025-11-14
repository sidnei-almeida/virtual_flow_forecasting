#!/usr/bin/env python3
"""
Script to retrain the LSTM model and save training history
"""

import pandas as pd
import numpy as np
import json
import os

# Force TensorFlow to use CPU only (disable GPU/CUDA)
# Must be set before importing TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings (0=all, 1=exclude info, 2=exclude info and warnings, 3=exclude all)

import tensorflow as tf
# Configure TensorFlow to use CPU only and suppress warnings
tf.config.set_visible_devices([], 'GPU')
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_data():
    """Load preprocessed data"""
    print("📊 Loading data...")
    train_df = pd.read_csv('data/train_data_scaled_manual.csv')
    test_df = pd.read_csv('data/test_data_scaled_manual.csv')
    print(f"✅ Data loaded: {len(train_df)} train, {len(test_df)} test")
    return train_df, test_df

def prepare_data(train_df, test_df):
    """Prepare data for LSTM model"""
    features = ['pressure_1', 'pressure_2', 'pressure_3', 'pressure_4', 'pressure_5', 'pressure_6', 'pressure_7']
    target = 'liquid_flow_rate'
    
    # Separate features and target
    X_train = train_df[features].values
    y_train = train_df[target].values
    X_test = test_df[features].values
    y_test = test_df[target].values
    
    # Reshape for LSTM
    X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    
    print(f"✅ Data prepared: X_train {X_train_reshaped.shape}, X_test {X_test_reshaped.shape}")
    return X_train_reshaped, y_train, X_test_reshaped, y_test

def create_model():
    """Create the LSTM model"""
    print("🤖 Creating LSTM model...")
    model = Sequential()
    model.add(LSTM(50, input_shape=(1, 7)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(f"✅ Model created: {model.count_params()} parameters")
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    """Train the model and return history"""
    print("🏋️ Starting training...")
    
    # Train with history
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=72,
        validation_data=(X_test, y_test),
        verbose=2,
        shuffle=False
    )
    
    print("✅ Training completed!")
    return history

def save_model_and_history(model, history):
    """Save model and training history"""
    print("💾 Saving model and history...")
    
    # Save model
    model.save('model/meu_modelo_lstm.keras')
    print("✅ Model saved: model/meu_modelo_lstm.keras")
    
    # Save history as JSON
    history_dict = {
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss']
    }
    
    with open('model/training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)
    print("✅ History saved: model/training_history.json")
    
    return history_dict

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics"""
    print("📈 Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred = y_pred.flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    print("📊 Final metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²: {r2:.6f}")
    
    return metrics

def main():
    """Main function"""
    print("🚀 Starting LSTM model retraining with history...")
    print("=" * 60)
    
    try:
        # Load data
        train_df, test_df = load_data()
        
        # Prepare data
        X_train, y_train, X_test, y_test = prepare_data(train_df, test_df)
        
        # Create model
        model = create_model()
        
        # Train model
        history = train_model(model, X_train, y_train, X_test, y_test)
        
        # Save model and history
        history_dict = save_model_and_history(model, history)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Save metrics
        with open('model/model_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        print("✅ Metrics saved: model/model_metrics.json")
        
        print("=" * 60)
        print("🎉 Retraining completed successfully!")
        print("📁 Generated files:")
        print("  - model/meu_modelo_lstm.keras")
        print("  - model/training_history.json")
        print("  - model/model_metrics.json")
        
    except Exception as e:
        print(f"❌ Error during retraining: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 Model retrained successfully! The API will use the updated model.")
    else:
        print("\n⚠️  Check the errors above before continuing.")
