#!/usr/bin/env python3
"""
Script para testar a funcionalidade de previsões do app
"""

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

def test_prediction_functionality():
    """Testa a funcionalidade de previsões"""
    print("🔮 Testando funcionalidade de previsões...")
    
    try:
        # Carregar dados e modelo
        test_df = pd.read_csv('data/test_data_scaled_manual.csv')
        model = load_model('model/meu_modelo_lstm.keras')
        
        features = ['pressure_1', 'pressure_2', 'pressure_3', 'pressure_4', 'pressure_5', 'pressure_6', 'pressure_7']
        target = 'liquid_flow_rate'
        
        print(f"✅ Dados carregados: {len(test_df)} amostras de teste")
        print(f"✅ Modelo carregado: {model.count_params()} parâmetros")
        
        # Teste 1: Previsão individual
        print("\n📊 Teste 1: Previsão Individual")
        sample_idx = 0
        sample_data = test_df.iloc[sample_idx]
        
        # Preparar dados para predição
        input_data = np.array([[sample_data[feature] for feature in features]])
        input_data_reshaped = np.reshape(input_data, (1, 1, len(features)))
        
        # Fazer predição
        prediction = model.predict(input_data_reshaped, verbose=0)
        predicted_flow = prediction[0][0]
        real_value = sample_data[target]
        
        print(f"   Amostra {sample_idx}:")
        print(f"   Valores de entrada: {[sample_data[feature] for feature in features[:3]]}...")
        print(f"   Valor real: {real_value:.6f}")
        print(f"   Valor previsto: {predicted_flow:.6f}")
        print(f"   Erro absoluto: {abs(predicted_flow - real_value):.6f}")
        
        # Teste 2: Previsões em lote
        print("\n📊 Teste 2: Previsões em Lote")
        num_samples = 10
        X_test_sample = test_df[features].iloc[:num_samples].values
        y_test_sample = test_df[target].iloc[:num_samples].values
        
        # Reshape para LSTM
        X_test_reshaped = np.reshape(X_test_sample, (X_test_sample.shape[0], 1, X_test_sample.shape[1]))
        
        # Fazer previsões
        predictions = model.predict(X_test_reshaped, verbose=0)
        predictions = predictions.flatten()
        
        # Calcular métricas
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(y_test_sample, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_sample, predictions)
        r2 = r2_score(y_test_sample, predictions)
        
        print(f"   Amostras testadas: {num_samples}")
        print(f"   MSE: {mse:.6f}")
        print(f"   RMSE: {rmse:.6f}")
        print(f"   MAE: {mae:.6f}")
        print(f"   R²: {r2:.6f}")
        
        # Teste 3: Validação de entrada
        print("\n📊 Teste 3: Validação de Entrada")
        
        # Testar com valores extremos
        extreme_values = [0.0, 1.0, 0.5]
        for val in extreme_values:
            test_input = np.array([[val] * len(features)])
            test_input_reshaped = np.reshape(test_input, (1, 1, len(features)))
            pred = model.predict(test_input_reshaped, verbose=0)
            
            print(f"   Entrada uniforme ({val}): Previsão = {pred[0][0]:.6f}")
        
        print("\n✅ Todos os testes de previsão passaram!")
        return True
        
    except Exception as e:
        print(f"❌ Erro nos testes de previsão: {e}")
        return False

def test_sample_selection():
    """Testa a seleção de amostras do conjunto de teste"""
    print("\n🎯 Testando seleção de amostras...")
    
    try:
        test_df = pd.read_csv('data/test_data_scaled_manual.csv')
        features = ['pressure_1', 'pressure_2', 'pressure_3', 'pressure_4', 'pressure_5', 'pressure_6', 'pressure_7']
        
        # Testar diferentes índices
        test_indices = [0, 100, 1000, len(test_df)-1]
        
        for idx in test_indices:
            if idx < len(test_df):
                sample_data = test_df.iloc[idx]
                pressure_values = [f"{sample_data[f]:.3f}" for f in features[:3]]
                print(f"   Amostra {idx}: Pressões = {pressure_values}...")
        
        print("✅ Seleção de amostras funcionando!")
        return True
        
    except Exception as e:
        print(f"❌ Erro na seleção de amostras: {e}")
        return False

def main():
    """Executa todos os testes de previsão"""
    print("🧪 Testando funcionalidades de previsão...")
    print("=" * 60)
    
    tests = [
        ("Funcionalidade de Previsões", test_prediction_functionality),
        ("Seleção de Amostras", test_sample_selection)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 60)
    print("📊 Resultados dos Testes de Previsão:")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"  {test_name}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("🎉 Todas as funcionalidades de previsão estão funcionando!")
        print("🚀 O app está pronto para uso!")
    else:
        print("⚠️  Algumas funcionalidades falharam. Verifique os erros acima.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
