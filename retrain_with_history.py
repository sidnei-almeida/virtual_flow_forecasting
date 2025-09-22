#!/usr/bin/env python3
"""
Script para retreinar o modelo LSTM e salvar o histórico de perda
"""

import pandas as pd
import numpy as np
import json
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_data():
    """Carrega os dados pré-processados"""
    print("📊 Carregando dados...")
    train_df = pd.read_csv('data/train_data_scaled_manual.csv')
    test_df = pd.read_csv('data/test_data_scaled_manual.csv')
    print(f"✅ Dados carregados: {len(train_df)} treino, {len(test_df)} teste")
    return train_df, test_df

def prepare_data(train_df, test_df):
    """Prepara os dados para o modelo LSTM"""
    features = ['pressure_1', 'pressure_2', 'pressure_3', 'pressure_4', 'pressure_5', 'pressure_6', 'pressure_7']
    target = 'liquid_flow_rate'
    
    # Separar features e target
    X_train = train_df[features].values
    y_train = train_df[target].values
    X_test = test_df[features].values
    y_test = test_df[target].values
    
    # Reshape para LSTM
    X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    
    print(f"✅ Dados preparados: X_train {X_train_reshaped.shape}, X_test {X_test_reshaped.shape}")
    return X_train_reshaped, y_train, X_test_reshaped, y_test

def create_model():
    """Cria o modelo LSTM"""
    print("🤖 Criando modelo LSTM...")
    model = Sequential()
    model.add(LSTM(50, input_shape=(1, 7)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(f"✅ Modelo criado: {model.count_params()} parâmetros")
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    """Treina o modelo e retorna o histórico"""
    print("🏋️ Iniciando treinamento...")
    
    # Treinar com histórico
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=72,
        validation_data=(X_test, y_test),
        verbose=2,
        shuffle=False
    )
    
    print("✅ Treinamento concluído!")
    return history

def save_model_and_history(model, history):
    """Salva o modelo e o histórico de treinamento"""
    print("💾 Salvando modelo e histórico...")
    
    # Salvar modelo
    model.save('model/meu_modelo_lstm.keras')
    print("✅ Modelo salvo: model/meu_modelo_lstm.keras")
    
    # Salvar histórico como JSON
    history_dict = {
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss']
    }
    
    with open('model/training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)
    print("✅ Histórico salvo: model/training_history.json")
    
    return history_dict

def evaluate_model(model, X_test, y_test):
    """Avalia o modelo e retorna métricas"""
    print("📈 Avaliando modelo...")
    
    # Fazer previsões
    y_pred = model.predict(X_test, verbose=0)
    y_pred = y_pred.flatten()
    
    # Calcular métricas
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
    
    print("📊 Métricas finais:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²: {r2:.6f}")
    
    return metrics

def main():
    """Função principal"""
    print("🚀 Iniciando retreinamento do modelo LSTM com histórico...")
    print("=" * 60)
    
    try:
        # Carregar dados
        train_df, test_df = load_data()
        
        # Preparar dados
        X_train, y_train, X_test, y_test = prepare_data(train_df, test_df)
        
        # Criar modelo
        model = create_model()
        
        # Treinar modelo
        history = train_model(model, X_train, y_train, X_test, y_test)
        
        # Salvar modelo e histórico
        history_dict = save_model_and_history(model, history)
        
        # Avaliar modelo
        metrics = evaluate_model(model, X_test, y_test)
        
        # Salvar métricas
        with open('model/model_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        print("✅ Métricas salvas: model/model_metrics.json")
        
        print("=" * 60)
        print("🎉 Retreinamento concluído com sucesso!")
        print("📁 Arquivos gerados:")
        print("  - model/meu_modelo_lstm.keras")
        print("  - model/training_history.json")
        print("  - model/model_metrics.json")
        
    except Exception as e:
        print(f"❌ Erro durante o retreinamento: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 Agora você pode executar o app Streamlit atualizado!")
        print("   streamlit run app.py")
    else:
        print("\n⚠️  Verifique os erros acima antes de continuar.")
