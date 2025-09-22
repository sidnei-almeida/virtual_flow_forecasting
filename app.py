import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import warnings
from streamlit_option_menu import option_menu
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="🌊 Previsão de Vazão de Líquido - LSTM",
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
    <p>Modelo LSTM Avançado para Análise de Fluxo Multifásico</p>
</div>
""", unsafe_allow_html=True)

# Função para carregar dados
@st.cache_data
def load_data():
    """Carrega os dados de treino, teste e originais"""
    try:
        train_df = pd.read_csv('data/train_data_scaled_manual.csv')
        test_df = pd.read_csv('data/test_data_scaled_manual.csv')
        raw_df = pd.read_csv('data/riser_pq_uni.csv')
        return train_df, test_df, raw_df
    except FileNotFoundError as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None, None, None

# Função para carregar modelo
@st.cache_resource
def load_lstm_model():
    """Carrega o modelo LSTM treinado"""
    try:
        model = load_model('model/meu_modelo_lstm.keras')
        return model
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None

# Função para carregar histórico de treinamento
@st.cache_data
def load_training_history():
    """Carrega o histórico de treinamento"""
    try:
        with open('model/training_history.json', 'r') as f:
            history = json.load(f)
        return history
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Erro ao carregar histórico: {e}")
        return None

# Função para carregar métricas do modelo
@st.cache_data
def load_model_metrics():
    """Carrega as métricas salvas do modelo"""
    try:
        with open('model/model_metrics.json', 'r') as f:
            metrics = json.load(f)
        return metrics
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Erro ao carregar métricas: {e}")
        return None

# Carregar dados e modelo
train_df, test_df, raw_df = load_data()
model = load_lstm_model()
training_history = load_training_history()
model_metrics = load_model_metrics()

if train_df is not None and test_df is not None and raw_df is not None and model is not None:
    
    # Menu de navegação moderno com streamlit-option-menu
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
            st.metric("📈 Modelo", "Ativo", delta="Online")
        with col2:
            st.metric("💾 Dados", "35.3K", delta="Carregados")
        
        st.markdown("### 🎯 Performance")
        if model_metrics is not None:
            st.metric("🎯 R² Score", f"{model_metrics['r2']:.3f}", delta="Excelente")
            st.metric("📉 RMSE", f"{model_metrics['rmse']:.4f}", delta="Baixo")
        
        # Informações do modelo
        st.markdown("### 🤖 Modelo LSTM")
        st.info(f"""
        **Arquitetura:** LSTM(50) + Dense(1)
        
        **Parâmetros:** {model.count_params():,}
        
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
    
    # Definir features e target
    features = ['pressure_1', 'pressure_2', 'pressure_3', 'pressure_4', 'pressure_5', 'pressure_6', 'pressure_7']
    target = 'liquid_flow_rate'
    
    if page == "Visão Geral":
        st.header("📋 Visão Geral do Projeto")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Total de Amostras", f"{len(raw_df):,}")
            st.metric("🏋️ Amostras de Treino", f"{len(train_df):,}")
        
        with col2:
            st.metric("🧪 Amostras de Teste", f"{len(test_df):,}")
            st.metric("⏱️ Duração (segundos)", f"{raw_df['Time (sec)'].max():.1f}")
        
        with col3:
            st.metric("🔢 Features", "7 (Pressões)")
            st.metric("🎯 Target", "Vazão de Líquido")
        
        st.markdown("---")
        
        # Estatísticas dos dados
        st.subheader("📊 Estatísticas Descritivas")
        
        # Estatísticas das pressões
        pressure_cols = [col for col in raw_df.columns if 'Pressure' in col]
        pressure_stats = raw_df[pressure_cols].describe()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Pressões (bar)**")
            st.dataframe(pressure_stats.round(4))
        
        with col2:
            st.write("**Vazões (kg/s)**")
            flow_stats = raw_df[['Gas mass rate (kg/sec)', 'Liquid mass rate (kg/sec)']].describe()
            st.dataframe(flow_stats.round(4))
        
        # Informações do modelo
        st.subheader("🤖 Informações do Modelo LSTM")
        
        model_info = {
            "Arquitetura": "LSTM + Dense",
            "Neurônios LSTM": "50",
            "Épocas de Treinamento": "50",
            "Batch Size": "72",
            "Função de Perda": "Mean Squared Error",
            "Otimizador": "Adam"
        }
        
        for key, value in model_info.items():
            st.write(f"**{key}:** {value}")
    
    elif page == "Visualização":
        st.header("📊 Visualização e Análise dos Dados")
        
        # Seleção de tipo de dados
        data_type = st.selectbox(
            "Selecione o tipo de dados para visualizar:",
            ["Dados Originais", "Dados de Treino", "Dados de Teste"]
        )
        
        if data_type == "Dados Originais":
            df = raw_df
            time_col = 'Time (sec)'
            pressure_cols = [col for col in df.columns if 'Pressure' in col]
        elif data_type == "Dados de Treino":
            df = train_df
            time_col = 'time'
            pressure_cols = features
        else:
            df = test_df
            time_col = 'time'
            pressure_cols = features
        
        # Visualização temporal das pressões
        st.subheader("📈 Evolução Temporal das Pressões")
        
        # Seleção de pressões para visualizar
        selected_pressures = st.multiselect(
            "Selecione as pressões para visualizar:",
            pressure_cols,
            default=pressure_cols[:4]  # Mostrar as primeiras 4 por padrão
        )
        
        if selected_pressures:
            fig = go.Figure()
            
            for pressure in selected_pressures:
                fig.add_trace(go.Scatter(
                    x=df[time_col],
                    y=df[pressure],
                    mode='lines',
                    name=pressure,
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title="Evolução Temporal das Pressões",
                xaxis_title="Tempo (segundos)",
                yaxis_title="Pressão (bar)",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Visualização da vazão de líquido
        st.subheader("💧 Vazão de Líquido ao Longo do Tempo")
        
        if data_type == "Dados Originais":
            liquid_col = 'Liquid mass rate (kg/sec)'
        else:
            liquid_col = 'liquid_flow_rate'
        
        fig_liquid = go.Figure()
        fig_liquid.add_trace(go.Scatter(
            x=df[time_col],
            y=df[liquid_col],
            mode='lines',
            name='Vazão de Líquido',
            line=dict(color='blue', width=2)
        ))
        
        fig_liquid.update_layout(
            title="Vazão de Líquido ao Longo do Tempo",
            xaxis_title="Tempo (segundos)",
            yaxis_title="Vazão (kg/s)" if data_type == "Dados Originais" else "Vazão (escalonada)",
            height=400
        )
        
        st.plotly_chart(fig_liquid, use_container_width=True)
        
        # Matriz de correlação
        st.subheader("🔗 Matriz de Correlação")
        
        if data_type == "Dados Originais":
            corr_data = df[pressure_cols + ['Liquid mass rate (kg/sec)']]
        else:
            corr_data = df[features + [target]]
        
        corr_matrix = corr_data.corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Matriz de Correlação entre Features e Target"
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Distribuição das variáveis
        st.subheader("📊 Distribuições das Variáveis")
        
        selected_var = st.selectbox(
            "Selecione uma variável para ver sua distribuição:",
            pressure_cols + ([liquid_col])
        )
        
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=df[selected_var],
            nbinsx=50,
            name=selected_var
        ))
        
        fig_dist.update_layout(
            title=f"Distribuição de {selected_var}",
            xaxis_title=selected_var,
            yaxis_title="Frequência",
            height=400
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
    
    elif page == "Previsões":
        st.header("🔮 Fazer Previsões com o Modelo LSTM")
        
        st.markdown("Insira os valores das pressões para prever a vazão de líquido:")
        
        # Interface para entrada de dados
        col1, col2 = st.columns(2)
        
        with col2:
            st.subheader("⚙️ Configurações")
            
            # Opção para usar dados de exemplo
            use_sample = st.checkbox("Usar dados de exemplo do conjunto de teste")
            
            if use_sample:
                sample_idx = st.slider(
                    "Índice da amostra:",
                    min_value=0,
                    max_value=len(test_df)-1,
                    value=0,
                    step=1
                )
                
                # Carregar dados da amostra selecionada
                sample_data = test_df.iloc[sample_idx]
                
                st.write("**Dados da Amostra Selecionada:**")
                for i, feature in enumerate(features):
                    st.write(f"Pressão {i+1}: {sample_data[feature]:.4f}")
        
        with col1:
            st.subheader("📊 Valores das Pressões")
            
            # Inicializar inputs com valores padrão ou da amostra
            pressure_inputs = {}
            
            if use_sample:
                # Usar dados da amostra como valores padrão
                for i, feature in enumerate(features):
                    pressure_inputs[feature] = st.number_input(
                        f"Pressão {i+1}",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(sample_data[feature]),
                        step=0.01,
                        format="%.4f",
                        key=f"pressure_{i+1}_sample"
                    )
            else:
                # Usar valores padrão
                for i, feature in enumerate(features):
                    pressure_inputs[feature] = st.number_input(
                        f"Pressão {i+1}",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.01,
                        format="%.4f",
                        key=f"pressure_{i+1}_manual"
                    )
        
        # Botão para fazer previsão
        if st.button("🔮 Fazer Previsão", type="primary"):
            # Preparar dados para predição
            input_data = np.array([[pressure_inputs[feature] for feature in features]])
            input_data_reshaped = np.reshape(input_data, (1, 1, len(features)))
            
            # Fazer predição
            prediction = model.predict(input_data_reshaped, verbose=0)
            predicted_flow = prediction[0][0]
            
            # Mostrar resultado
            st.success(f"**Vazão de Líquido Prevista:** {predicted_flow:.6f}")
            
            # Se estiver usando dados de exemplo, mostrar valor real
            if use_sample:
                real_value = test_df.iloc[sample_idx][target]
                st.info(f"**Vazão de Líquido Real:** {real_value:.6f}")
                
                # Calcular erro
                error = abs(predicted_flow - real_value)
                error_percent = (error / real_value) * 100 if real_value != 0 else 0
                
                st.metric("Erro Absoluto", f"{error:.6f}")
                st.metric("Erro Percentual", f"{error_percent:.2f}%")
        
        # Visualização de múltiplas previsões
        st.subheader("📊 Previsões em Lote")
        
        num_predictions = st.slider(
            "Número de amostras para prever:",
            min_value=10,
            max_value=1000,
            value=100,
            step=10
        )
        
        if st.button("Gerar Previsões em Lote"):
            # Usar dados de teste para previsões em lote
            X_test_sample = test_df[features].iloc[:num_predictions].values
            y_test_sample = test_df[target].iloc[:num_predictions].values
            
            # Reshape para LSTM
            X_test_reshaped = np.reshape(X_test_sample, (X_test_sample.shape[0], 1, X_test_sample.shape[1]))
            
            # Fazer previsões
            predictions = model.predict(X_test_reshaped, verbose=0)
            predictions = predictions.flatten()
            
            # Criar gráfico de comparação
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=list(range(len(predictions))),
                y=y_test_sample,
                mode='lines',
                name='Valores Reais',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=list(range(len(predictions))),
                y=predictions,
                mode='lines',
                name='Previsões',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title=f"Comparação: Valores Reais vs Previsões ({num_predictions} amostras)",
                xaxis_title="Índice da Amostra",
                yaxis_title="Vazão de Líquido (escalonada)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Métricas de performance
            mse = mean_squared_error(y_test_sample, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_sample, predictions)
            r2 = r2_score(y_test_sample, predictions)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("MSE", f"{mse:.6f}")
            with col2:
                st.metric("RMSE", f"{rmse:.6f}")
            with col3:
                st.metric("MAE", f"{mae:.6f}")
            with col4:
                st.metric("R²", f"{r2:.6f}")
    
    elif page == "Avaliação":
        st.header("📈 Avaliação Completa do Modelo")
        
        # Carregar dados de teste
        X_test = test_df[features].values
        y_test = test_df[target].values
        
        # Reshape para LSTM
        X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        
        # Fazer previsões
        y_pred = model.predict(X_test_reshaped, verbose=0)
        y_pred = y_pred.flatten()
        
        # Métricas de avaliação
        st.subheader("📊 Métricas de Performance")
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("MSE", f"{mse:.6f}", help="Mean Squared Error")
        with col2:
            st.metric("RMSE", f"{rmse:.6f}", help="Root Mean Squared Error")
        with col3:
            st.metric("MAE", f"{mae:.6f}", help="Mean Absolute Error")
        with col4:
            st.metric("R²", f"{r2:.6f}", help="Coeficiente de Determinação")
        
        # Gráfico de dispersão
        st.subheader("🎯 Gráfico de Dispersão: Real vs Previsto")
        
        fig_scatter = go.Figure()
        
        fig_scatter.add_trace(go.Scatter(
            x=y_test,
            y=y_pred,
            mode='markers',
            marker=dict(
                color='blue',
                size=4,
                opacity=0.6
            ),
            name='Previsões'
        ))
        
        # Linha de referência (y = x)
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Linha de Referência (y = x)'
        ))
        
        fig_scatter.update_layout(
            title="Valores Reais vs Previsões",
            xaxis_title="Valores Reais",
            yaxis_title="Valores Previstos",
            height=500
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Gráfico temporal
        st.subheader("⏰ Comparação Temporal")
        
        # Selecionar uma janela de tempo para visualizar
        window_size = st.slider(
            "Tamanho da janela temporal:",
            min_value=100,
            max_value=len(y_test),
            value=1000,
            step=100
        )
        
        start_idx = st.slider(
            "Índice inicial:",
            min_value=0,
            max_value=len(y_test) - window_size,
            value=0,
            step=100
        )
        
        end_idx = start_idx + window_size
        
        fig_time = go.Figure()
        
        fig_time.add_trace(go.Scatter(
            x=list(range(start_idx, end_idx)),
            y=y_test[start_idx:end_idx],
            mode='lines',
            name='Valores Reais',
            line=dict(color='blue', width=2)
        ))
        
        fig_time.add_trace(go.Scatter(
            x=list(range(start_idx, end_idx)),
            y=y_pred[start_idx:end_idx],
            mode='lines',
            name='Previsões',
            line=dict(color='red', width=2)
        ))
        
        fig_time.update_layout(
            title=f"Comparação Temporal (Amostras {start_idx}-{end_idx})",
            xaxis_title="Índice da Amostra",
            yaxis_title="Vazão de Líquido (escalonada)",
            height=500
        )
        
        st.plotly_chart(fig_time, use_container_width=True)
        
        # Análise de resíduos
        st.subheader("📉 Análise de Resíduos")
        
        residuals = y_test - y_pred
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histograma dos resíduos
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=residuals,
                nbinsx=50,
                name='Resíduos'
            ))
            
            fig_hist.update_layout(
                title="Distribuição dos Resíduos",
                xaxis_title="Resíduos",
                yaxis_title="Frequência",
                height=400
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Resíduos vs Valores Previstos
            fig_resid = go.Figure()
            fig_resid.add_trace(go.Scatter(
                x=y_pred,
                y=residuals,
                mode='markers',
                marker=dict(
                    color='green',
                    size=4,
                    opacity=0.6
                ),
                name='Resíduos'
            ))
            
            # Linha horizontal em y=0
            fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
            
            fig_resid.update_layout(
                title="Resíduos vs Valores Previstos",
                xaxis_title="Valores Previstos",
                yaxis_title="Resíduos",
                height=400
            )
            
            st.plotly_chart(fig_resid, use_container_width=True)
        
        # Estatísticas dos resíduos
        st.subheader("📊 Estatísticas dos Resíduos")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Média dos Resíduos", f"{np.mean(residuals):.6f}")
        with col2:
            st.metric("Desvio Padrão", f"{np.std(residuals):.6f}")
        with col3:
            st.metric("Máximo Absoluto", f"{np.max(np.abs(residuals)):.6f}")
    
    elif page == "Treinamento":
        st.header("🏋️ Análise do Histórico de Treinamento")
        
        if training_history is not None:
            # Gráfico das curvas de perda
            st.subheader("📉 Curvas de Perda durante o Treinamento")
            
            epochs = list(range(1, len(training_history['loss']) + 1))
            
            fig_loss = go.Figure()
            
            fig_loss.add_trace(go.Scatter(
                x=epochs,
                y=training_history['loss'],
                mode='lines',
                name='Perda de Treino',
                line=dict(color='blue', width=3)
            ))
            
            fig_loss.add_trace(go.Scatter(
                x=epochs,
                y=training_history['val_loss'],
                mode='lines',
                name='Perda de Validação',
                line=dict(color='red', width=3)
            ))
            
            fig_loss.update_layout(
                title="Evolução da Perda durante o Treinamento",
                xaxis_title="Época",
                yaxis_title="Perda (MSE)",
                hovermode='x unified',
                height=500,
                legend=dict(x=0.02, y=0.98)
            )
            
            st.plotly_chart(fig_loss, use_container_width=True)
            
            # Análise do treinamento
            st.subheader("📊 Análise do Treinamento")
            
            col1, col2, col3, col4 = st.columns(4)
            
            final_train_loss = training_history['loss'][-1]
            final_val_loss = training_history['val_loss'][-1]
            initial_train_loss = training_history['loss'][0]
            initial_val_loss = training_history['val_loss'][0]
            
            with col1:
                st.metric("Perda Inicial (Treino)", f"{initial_train_loss:.6f}")
                st.metric("Perda Final (Treino)", f"{final_train_loss:.6f}")
            
            with col2:
                st.metric("Perda Inicial (Validação)", f"{initial_val_loss:.6f}")
                st.metric("Perda Final (Validação)", f"{final_val_loss:.6f}")
            
            with col3:
                improvement_train = ((initial_train_loss - final_train_loss) / initial_train_loss) * 100
                improvement_val = ((initial_val_loss - final_val_loss) / initial_val_loss) * 100
                st.metric("Melhoria Treino", f"{improvement_train:.1f}%")
                st.metric("Melhoria Validação", f"{improvement_val:.1f}%")
            
            with col4:
                overfitting = final_val_loss - final_train_loss
                st.metric("Diferença Final", f"{overfitting:.6f}")
                if overfitting > 0.001:
                    st.warning("⚠️ Possível overfitting")
                else:
                    st.success("✅ Boa generalização")
            
            # Gráfico de melhorias
            st.subheader("📈 Análise de Convergência")
            
            # Calcular melhorias por época
            train_improvements = []
            val_improvements = []
            
            for i in range(1, len(training_history['loss'])):
                train_improvement = ((training_history['loss'][i-1] - training_history['loss'][i]) / training_history['loss'][i-1]) * 100
                val_improvement = ((training_history['val_loss'][i-1] - training_history['val_loss'][i]) / training_history['val_loss'][i-1]) * 100
                
                train_improvements.append(train_improvement)
                val_improvements.append(val_improvement)
            
            fig_improvements = go.Figure()
            
            fig_improvements.add_trace(go.Scatter(
                x=list(range(2, len(training_history['loss']) + 1)),
                y=train_improvements,
                mode='lines',
                name='Melhoria Treino (%)',
                line=dict(color='blue', width=2)
            ))
            
            fig_improvements.add_trace(go.Scatter(
                x=list(range(2, len(training_history['loss']) + 1)),
                y=val_improvements,
                mode='lines',
                name='Melhoria Validação (%)',
                line=dict(color='red', width=2)
            ))
            
            fig_improvements.update_layout(
                title="Melhoria Percentual por Época",
                xaxis_title="Época",
                yaxis_title="Melhoria (%)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig_improvements, use_container_width=True)
            
            # Estatísticas do treinamento
            st.subheader("📋 Estatísticas Detalhadas")
            
            # Encontrar melhor época
            best_epoch_val = np.argmin(training_history['val_loss']) + 1
            best_epoch_train = np.argmin(training_history['loss']) + 1
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Melhores Resultados:**")
                st.write(f"• Melhor época (validação): {best_epoch_val}")
                st.write(f"• Melhor época (treino): {best_epoch_train}")
                st.write(f"• Menor perda validação: {min(training_history['val_loss']):.6f}")
                st.write(f"• Menor perda treino: {min(training_history['loss']):.6f}")
            
            with col2:
                st.write("**Convergência:**")
                # Calcular estabilidade nas últimas 10 épocas
                last_10_train = training_history['loss'][-10:]
                last_10_val = training_history['val_loss'][-10:]
                
                stability_train = np.std(last_10_train)
                stability_val = np.std(last_10_val)
                
                st.write(f"• Estabilidade treino (últimas 10): {stability_train:.8f}")
                st.write(f"• Estabilidade validação (últimas 10): {stability_val:.8f}")
                st.write(f"• Total de épocas: {len(training_history['loss'])}")
                
                if stability_train < 0.0001 and stability_val < 0.0001:
                    st.success("✅ Modelo convergiu bem")
                elif stability_train < 0.001 and stability_val < 0.001:
                    st.info("ℹ️ Modelo estável")
                else:
                    st.warning("⚠️ Modelo pode precisar de mais épocas")
            
            # Gráfico de comparação logarítmica
            st.subheader("📊 Visualização Logarítmica")
            
            fig_log = go.Figure()
            
            fig_log.add_trace(go.Scatter(
                x=epochs,
                y=training_history['loss'],
                mode='lines',
                name='Perda de Treino',
                line=dict(color='blue', width=3)
            ))
            
            fig_log.add_trace(go.Scatter(
                x=epochs,
                y=training_history['val_loss'],
                mode='lines',
                name='Perda de Validação',
                line=dict(color='red', width=3)
            ))
            
            fig_log.update_layout(
                title="Curvas de Perda (Escala Logarítmica)",
                xaxis_title="Época",
                yaxis_title="Perda (MSE) - Escala Log",
                yaxis_type="log",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig_log, use_container_width=True)
            
        else:
            st.warning("⚠️ Histórico de treinamento não encontrado. Execute o script de retreinamento primeiro.")
            
            if st.button("🔄 Executar Retreinamento"):
                st.info("Para retreinar o modelo com histórico, execute:")
                st.code("python retrain_with_history.py", language="bash")
    
    elif page == "Configurações":
        st.header("⚙️ Configurações e Informações Técnicas")
        
        # Informações do modelo
        st.subheader("🤖 Informações do Modelo")
        
        model_config = {
            "Nome do Arquivo": "meu_modelo_lstm.keras",
            "Tipo de Modelo": "LSTM Neural Network",
            "Camadas": "LSTM(50) + Dense(1)",
            "Parâmetros Treináveis": f"{model.count_params():,}",
            "Função de Ativação (LSTM)": "tanh",
            "Função de Ativação (Saída)": "linear",
            "Função de Perda": "mean_squared_error",
            "Otimizador": "adam"
        }
        
        for key, value in model_config.items():
            st.write(f"**{key}:** {value}")
        
        # Informações dos dados
        st.subheader("📊 Informações dos Dados")
        
        data_info = {
            "Dataset Original": "riser_pq_uni.csv",
            "Total de Amostras": f"{len(raw_df):,}",
            "Features de Entrada": "7 (pressões)",
            "Variável Alvo": "liquid_flow_rate",
            "Divisão Treino/Teste": "80% / 20%",
            "Escalonamento": "MinMaxScaler (0-1)",
            "Tipo de Problema": "Regressão"
        }
        
        for key, value in data_info.items():
            st.write(f"**{key}:** {value}")
        
        # Performance do modelo
        st.subheader("📈 Performance do Modelo")
        
        if model_metrics is not None:
            # Usar métricas salvas
            performance_metrics = {
                "MSE": f"{model_metrics['mse']:.6f}",
                "RMSE": f"{model_metrics['rmse']:.6f}",
                "MAE": f"{model_metrics['mae']:.6f}",
                "R²": f"{model_metrics['r2']:.6f}"
            }
        else:
            # Calcular métricas em tempo real se não estiverem salvas
            X_test = test_df[features].values
            y_test = test_df[target].values
            X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
            y_pred = model.predict(X_test_reshaped, verbose=0).flatten()
            
            performance_metrics = {
                "MSE": f"{mean_squared_error(y_test, y_pred):.6f}",
                "RMSE": f"{np.sqrt(mean_squared_error(y_test, y_pred)):.6f}",
                "MAE": f"{mean_absolute_error(y_test, y_pred):.6f}",
                "R²": f"{r2_score(y_test, y_pred):.6f}"
            }
        
        for key, value in performance_metrics.items():
            st.write(f"**{key}:** {value}")
        
        # Informações adicionais do treinamento
        if training_history is not None:
            st.subheader("🏋️ Informações do Treinamento")
            
            training_info = {
                "Total de Épocas": len(training_history['loss']),
                "Perda Final (Treino)": f"{training_history['loss'][-1]:.6f}",
                "Perda Final (Validação)": f"{training_history['val_loss'][-1]:.6f}",
                "Melhor Época (Validação)": np.argmin(training_history['val_loss']) + 1,
                "Menor Perda Validação": f"{min(training_history['val_loss']):.6f}"
            }
            
            for key, value in training_info.items():
                st.write(f"**{key}:** {value}")
        
        # Download de dados
        st.subheader("💾 Download de Dados")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Download Dados de Treino"):
                csv = train_df.to_csv(index=False)
                st.download_button(
                    label="Baixar CSV",
                    data=csv,
                    file_name="train_data_scaled_manual.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📥 Download Dados de Teste"):
                csv = test_df.to_csv(index=False)
                st.download_button(
                    label="Baixar CSV",
                    data=csv,
                    file_name="test_data_scaled_manual.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("📥 Download Dados Originais"):
                csv = raw_df.to_csv(index=False)
                st.download_button(
                    label="Baixar CSV",
                    data=csv,
                    file_name="riser_pq_uni.csv",
                    mime="text/csv"
                )

else:
    st.error("❌ Erro ao carregar dados ou modelo. Verifique se os arquivos estão no local correto.")
    st.info("📁 Estrutura esperada:\n- `data/train_data_scaled_manual.csv`\n- `data/test_data_scaled_manual.csv`\n- `data/riser_pq_uni.csv`\n- `model/meu_modelo_lstm.keras`")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🚀 <strong>Sistema de Previsão de Vazão de Líquido</strong> - Modelo LSTM</p>
        <p>Desenvolvido com Streamlit, TensorFlow/Keras e Plotly</p>
    </div>
    """,
    unsafe_allow_html=True
)
