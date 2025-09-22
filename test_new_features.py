#!/usr/bin/env python3
"""
Script para testar as novas funcionalidades do app Streamlit
"""

import json
import os
import numpy as np

def test_training_history():
    """Testa se o histórico de treinamento pode ser carregado"""
    print("🏋️ Testando histórico de treinamento...")
    
    try:
        with open('model/training_history.json', 'r') as f:
            history = json.load(f)
        
        print(f"✅ Histórico carregado: {len(history['loss'])} épocas")
        print(f"   Perda inicial (treino): {history['loss'][0]:.6f}")
        print(f"   Perda final (treino): {history['loss'][-1]:.6f}")
        print(f"   Perda inicial (validação): {history['val_loss'][0]:.6f}")
        print(f"   Perda final (validação): {history['val_loss'][-1]:.6f}")
        
        return True
        
    except FileNotFoundError:
        print("❌ Arquivo training_history.json não encontrado")
        return False
    except Exception as e:
        print(f"❌ Erro ao carregar histórico: {e}")
        return False

def test_model_metrics():
    """Testa se as métricas do modelo podem ser carregadas"""
    print("📊 Testando métricas do modelo...")
    
    try:
        with open('model/model_metrics.json', 'r') as f:
            metrics = json.load(f)
        
        print(f"✅ Métricas carregadas:")
        print(f"   MSE: {metrics['mse']:.6f}")
        print(f"   RMSE: {metrics['rmse']:.6f}")
        print(f"   MAE: {metrics['mae']:.6f}")
        print(f"   R²: {metrics['r2']:.6f}")
        
        return True
        
    except FileNotFoundError:
        print("❌ Arquivo model_metrics.json não encontrado")
        return False
    except Exception as e:
        print(f"❌ Erro ao carregar métricas: {e}")
        return False

def test_training_analysis():
    """Testa análises do histórico de treinamento"""
    print("📈 Testando análises de treinamento...")
    
    try:
        with open('model/training_history.json', 'r') as f:
            history = json.load(f)
        
        # Calcular melhorias
        initial_train = history['loss'][0]
        final_train = history['loss'][-1]
        improvement_train = ((initial_train - final_train) / initial_train) * 100
        
        initial_val = history['val_loss'][0]
        final_val = history['val_loss'][-1]
        improvement_val = ((initial_val - final_val) / initial_val) * 100
        
        print(f"✅ Análises calculadas:")
        print(f"   Melhoria treino: {improvement_train:.1f}%")
        print(f"   Melhoria validação: {improvement_val:.1f}%")
        
        # Verificar overfitting
        overfitting = final_val - final_train
        print(f"   Diferença final: {overfitting:.6f}")
        
        if overfitting > 0.001:
            print("   ⚠️ Possível overfitting detectado")
        else:
            print("   ✅ Boa generalização")
        
        # Encontrar melhor época
        best_epoch_val = np.argmin(history['val_loss']) + 1
        print(f"   Melhor época (validação): {best_epoch_val}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro nas análises: {e}")
        return False

def test_plotly_imports():
    """Testa se as bibliotecas Plotly estão funcionando"""
    print("📊 Testando imports Plotly...")
    
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        
        # Criar um gráfico simples
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 4, 2], mode='lines'))
        
        print("✅ Plotly funcionando corretamente")
        return True
        
    except Exception as e:
        print(f"❌ Erro no Plotly: {e}")
        return False

def main():
    """Executa todos os testes das novas funcionalidades"""
    print("🧪 Testando novas funcionalidades do app Streamlit...")
    print("=" * 60)
    
    tests = [
        ("Histórico de Treinamento", test_training_history),
        ("Métricas do Modelo", test_model_metrics),
        ("Análises de Treinamento", test_training_analysis),
        ("Imports Plotly", test_plotly_imports)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}:")
        result = test_func()
        results.append((test_name, result))
        print()
    
    print("=" * 60)
    print("📊 Resultados dos Testes das Novas Funcionalidades:")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"  {test_name}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("🎉 Todas as novas funcionalidades estão funcionando!")
        print("🚀 Execute: streamlit run app.py")
        print("📋 Nova seção disponível: '🏋️ Histórico de Treinamento'")
    else:
        print("⚠️  Algumas funcionalidades falharam. Verifique os erros acima.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
