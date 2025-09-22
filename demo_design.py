#!/usr/bin/env python3
"""
Script de demonstração do novo design do app Streamlit
"""

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np

# Configuração da página
st.set_page_config(
    page_title="🌊 Demo - Design Moderno",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
def load_css():
    with open('.streamlit/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Header personalizado
st.markdown("""
<div class="main-header fade-in-up">
    <h1>🌊 Sistema de Previsão de Vazão de Líquido</h1>
    <p>Design Moderno com Tema Dark e Menu Elegante</p>
</div>
""", unsafe_allow_html=True)

# Menu de navegação moderno
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #00D4AA, #00A8CC); border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0; font-weight: 600;">🌊 Menu</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Menu principal
    page = option_menu(
        menu_title=None,
        options=["Visão Geral", "Visualização", "Previsões", "Avaliação", "Treinamento", "Configurações"],
        icons=["house", "bar-chart", "cpu", "graph-up", "activity", "gear"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#00D4AA", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#1E1E1E",
                "color": "#FAFAFA",
                "background-color": "transparent",
                "padding": "12px 15px",
                "border-radius": "8px",
                "transition": "all 0.3s ease"
            },
            "nav-link-selected": {
                "background-color": "#00D4AA",
                "color": "white",
                "font-weight": "600",
                "box-shadow": "0 4px 15px rgba(0, 212, 170, 0.3)"
            },
            "nav-link:hover": {
                "background-color": "#00A8CC",
                "color": "white",
                "transform": "translateX(5px)"
            }
        }
    )
    
    # Informações do sistema
    st.markdown("---")
    st.markdown("### 📊 Status do Sistema")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📈 Modelo", "✅ Ativo", delta="Online")
    with col2:
        st.metric("💾 Dados", "✅ Carregados", delta="35.3K")
    
    st.markdown("### 🎯 Performance")
    st.metric("🎯 R² Score", "0.934", delta="Excelente")
    st.metric("📉 RMSE", "0.0199", delta="Baixo")
    
    # Informações do modelo
    st.markdown("### 🤖 Modelo LSTM")
    st.info(f"""
    **Arquitetura:** LSTM(50) + Dense(1)
    
    **Parâmetros:** 11,651
    
    **Status:** ✅ Treinado
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #00D4AA; font-size: 0.9rem; padding: 1rem;">
        🌊 <strong>Sistema de Previsão</strong><br>
        <small>Powered by LSTM & Streamlit</small><br>
        <small style="color: #666;">v2.0 - Dark Theme</small>
    </div>
    """, unsafe_allow_html=True)

# Conteúdo principal baseado na página selecionada
if page == "Visão Geral":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.header("📋 Visão Geral do Projeto")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📊 Total de Amostras", "35,369", delta="Dataset Completo")
        st.metric("🏋️ Amostras de Treino", "28,295", delta="80%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🧪 Amostras de Teste", "7,074", delta="20%")
        st.metric("⏱️ Duração (segundos)", "2,440.3", delta="Temporal")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🔢 Features", "7 (Pressões)", delta="Sensores")
        st.metric("🎯 Target", "Vazão de Líquido", delta="Regressão")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Visualização":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.header("📊 Visualização e Análise dos Dados")
    
    # Dados de exemplo
    data = np.random.randn(100, 7)
    df = pd.DataFrame(data, columns=[f'Pressão {i+1}' for i in range(7)])
    
    st.subheader("📈 Gráfico de Exemplo")
    st.line_chart(df)
    
    st.subheader("📊 Estatísticas Descritivas")
    st.dataframe(df.describe())
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Previsões":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.header("🔮 Fazer Previsões com o Modelo LSTM")
    
    st.subheader("📊 Valores das Pressões")
    col1, col2 = st.columns(2)
    
    with col1:
        pressure_1 = st.number_input("Pressão 1", 0.0, 1.0, 0.5, 0.01)
        pressure_2 = st.number_input("Pressão 2", 0.0, 1.0, 0.6, 0.01)
        pressure_3 = st.number_input("Pressão 3", 0.0, 1.0, 0.4, 0.01)
        pressure_4 = st.number_input("Pressão 4", 0.0, 1.0, 0.7, 0.01)
    
    with col2:
        pressure_5 = st.number_input("Pressão 5", 0.0, 1.0, 0.3, 0.01)
        pressure_6 = st.number_input("Pressão 6", 0.0, 1.0, 0.8, 0.01)
        pressure_7 = st.number_input("Pressão 7", 0.0, 1.0, 0.2, 0.01)
    
    if st.button("🔮 Fazer Previsão", type="primary"):
        # Simulação de previsão
        prediction = 0.3 + 0.2 * (pressure_1 + pressure_2 + pressure_3) / 3
        st.success(f"**Vazão de Líquido Prevista:** {prediction:.6f}")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Avaliação":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.header("📈 Avaliação Completa do Modelo")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("MSE", "0.000397", delta="Baixo")
    with col2:
        st.metric("RMSE", "0.019931", delta="Excelente")
    with col3:
        st.metric("MAE", "0.008890", delta="Muito Baixo")
    with col4:
        st.metric("R²", "0.934", delta="Excelente")
    
    st.subheader("🎯 Gráfico de Exemplo")
    # Dados simulados
    real_values = np.random.normal(0.3, 0.1, 100)
    predicted_values = real_values + np.random.normal(0, 0.02, 100)
    
    chart_data = pd.DataFrame({
        'Real': real_values,
        'Previsto': predicted_values
    })
    
    st.line_chart(chart_data)
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Treinamento":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.header("🏋️ Análise do Histórico de Treinamento")
    
    st.subheader("📉 Curvas de Perda")
    
    # Dados simulados para o gráfico
    epochs = list(range(1, 51))
    train_loss = [0.011 - 0.0002*i + np.random.normal(0, 0.001) for i in range(50)]
    val_loss = [0.004 - 0.00008*i + np.random.normal(0, 0.0005) for i in range(50)]
    
    chart_data = pd.DataFrame({
        'Época': epochs,
        'Perda Treino': train_loss,
        'Perda Validação': val_loss
    })
    
    st.line_chart(chart_data.set_index('Época'))
    
    st.subheader("📊 Métricas de Treinamento")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Melhoria Treino", "89.8%", delta="Excelente")
        st.metric("Melhoria Validação", "89.6%", delta="Excelente")
    
    with col2:
        st.metric("Melhor Época", "50", delta="Convergência")
        st.metric("Overfitting", "Não", delta="✅ Boa Generalização")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Configurações":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.header("⚙️ Configurações e Informações Técnicas")
    
    st.subheader("🤖 Informações do Modelo")
    
    model_info = {
        "Nome do Arquivo": "meu_modelo_lstm.keras",
        "Tipo de Modelo": "LSTM Neural Network",
        "Camadas": "LSTM(50) + Dense(1)",
        "Parâmetros Treináveis": "11,651",
        "Função de Ativação (LSTM)": "tanh",
        "Função de Ativação (Saída)": "linear",
        "Função de Perda": "mean_squared_error",
        "Otimizador": "adam"
    }
    
    for key, value in model_info.items():
        st.write(f"**{key}:** {value}")
    
    st.subheader("📊 Informações dos Dados")
    
    data_info = {
        "Dataset Original": "riser_pq_uni.csv",
        "Total de Amostras": "35,369",
        "Features de Entrada": "7 (pressões)",
        "Variável Alvo": "liquid_flow_rate",
        "Divisão Treino/Teste": "80% / 20%",
        "Escalonamento": "MinMaxScaler (0-1)",
        "Tipo de Problema": "Regressão"
    }
    
    for key, value in data_info.items():
        st.write(f"**{key}:** {value}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <h3>🚀 Sistema de Previsão de Vazão de Líquido</h3>
    <p>Desenvolvido com Streamlit, TensorFlow/Keras e Plotly</p>
    <p><strong>Design Moderno v2.0</strong> - Tema Dark com Menu Elegante</p>
</div>
""", unsafe_allow_html=True)
