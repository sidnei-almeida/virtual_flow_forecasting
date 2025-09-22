#!/usr/bin/env python3
"""
Script de teste para verificar se o app Streamlit está funcionando corretamente
"""

import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def test_data_loading():
    """Testa se os dados podem ser carregados"""
    print("🔍 Testando carregamento de dados...")
    
    try:
        # Testar dados de treino
        train_df = pd.read_csv('data/train_data_scaled_manual.csv')
        print(f"✅ Dados de treino carregados: {train_df.shape}")
        
        # Testar dados de teste
        test_df = pd.read_csv('data/test_data_scaled_manual.csv')
        print(f"✅ Dados de teste carregados: {test_df.shape}")
        
        # Testar dados originais
        raw_df = pd.read_csv('data/riser_pq_uni.csv')
        print(f"✅ Dados originais carregados: {raw_df.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        return False

def test_model_loading():
    """Testa se o modelo pode ser carregado"""
    print("🤖 Testando carregamento do modelo...")
    
    try:
        model = load_model('model/meu_modelo_lstm.keras')
        print(f"✅ Modelo carregado: {model.count_params()} parâmetros")
        
        # Testar uma predição simples
        features = ['pressure_1', 'pressure_2', 'pressure_3', 'pressure_4', 'pressure_5', 'pressure_6', 'pressure_7']
        test_input = np.array([[0.5] * 7])  # Valores médios
        test_input_reshaped = np.reshape(test_input, (1, 1, 7))
        
        prediction = model.predict(test_input_reshaped, verbose=0)
        print(f"✅ Predição de teste: {prediction[0][0]:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return False

def test_imports():
    """Testa se todas as dependências podem ser importadas"""
    print("📦 Testando importações...")
    
    try:
        import streamlit as st
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import tensorflow as tf
        import plotly.graph_objects as go
        import plotly.express as px
        
        print("✅ Todas as dependências importadas com sucesso")
        return True
        
    except Exception as e:
        print(f"❌ Erro nas importações: {e}")
        return False

def main():
    """Executa todos os testes"""
    print("🧪 Iniciando testes do app Streamlit...")
    print("=" * 50)
    
    tests = [
        ("Importações", test_imports),
        ("Carregamento de Dados", test_data_loading),
        ("Carregamento do Modelo", test_model_loading)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}:")
        result = test_func()
        results.append((test_name, result))
        print()
    
    print("=" * 50)
    print("📊 Resultados dos Testes:")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"  {test_name}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("🎉 Todos os testes passaram! O app está pronto para uso.")
        print("🚀 Execute: streamlit run app.py")
    else:
        print("⚠️  Alguns testes falharam. Verifique os erros acima.")
        sys.exit(1)

if __name__ == "__main__":
    main()
