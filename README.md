# 🌊 Virtual Flow Forecasting

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Sistema Avançado de Previsão de Vazão Multifásica** utilizando Deep Learning para análise de fluxo em dutos industriais.

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Características](#-características)
- [Arquitetura do Projeto](#-arquitetura-do-projeto)
- [Dados](#-dados)
- [Modelo LSTM](#-modelo-lstm)
- [Aplicação Streamlit](#-aplicação-streamlit)
- [Instalação](#-instalação)
- [Uso](#-uso)
- [Resultados](#-resultados)
- [Contribuição](#-contribuição)
- [Licença](#-licença)

## 🎯 Visão Geral

O **Virtual Flow Forecasting** é um sistema de inteligência artificial que utiliza redes neurais LSTM (Long Short-Term Memory) para prever a vazão de líquidos em sistemas de dutos industriais. O projeto combina técnicas avançadas de machine learning com uma interface web interativa para análise e previsão de fluxo multifásico.

### 🎯 Objetivos

- **Previsão Precisa**: Desenvolver modelos de deep learning para previsão de vazão de líquidos
- **Análise em Tempo Real**: Interface web para análise interativa de dados
- **Visualização Avançada**: Gráficos dinâmicos e métricas de performance
- **Deploy Simplificado**: Aplicação web acessível via GitHub Pages

## ✨ Características

### 🤖 **Inteligência Artificial**
- **Rede Neural LSTM** otimizada para séries temporais
- **Preprocessamento avançado** com normalização MinMax
- **Métricas de avaliação** abrangentes (MSE, RMSE, MAE, R²)
- **Histórico de treinamento** com análise de convergência

### 📊 **Análise de Dados**
- **35,369 registros** de dados reais de sensores industriais
- **7 features** de pressão em diferentes posições do duto
- **Dados multifásicos** (gás e líquido) com alta resolução temporal
- **Visualização interativa** com Plotly

### 🌐 **Interface Web**
- **Design moderno** com tema dark e navegação intuitiva
- **Previsões em tempo real** com validação de entrada
- **Dashboard completo** com métricas e gráficos
- **Carregamento remoto** direto do GitHub

## 🏗️ Arquitetura do Projeto

```
virtual_flow_forecasting/
├── 📁 data/                          # Dados do projeto
│   ├── riser_pq_uni.csv             # Dados originais (35K registros)
│   ├── train_data_scaled_manual.csv  # Dados de treino normalizados
│   └── test_data_scaled_manual.csv   # Dados de teste normalizados
├── 📁 model/                         # Modelos e métricas
│   ├── meu_modelo_lstm.keras        # Modelo LSTM treinado
│   ├── training_history.json        # Histórico de treinamento
│   └── model_metrics.json           # Métricas de avaliação
├── 📁 notebooks/                     # Jupyter Notebooks
│   ├── 1. Data Pre-Processing.ipynb # Preprocessamento de dados
│   └── 2. LSTM Model Training.ipynb # Treinamento do modelo
├── 📁 .streamlit/                    # Configurações do Streamlit
│   ├── config.toml                  # Configuração do tema
│   └── style.css                    # Estilos customizados
├── app.py                           # Aplicação Streamlit principal
├── requirements.txt                 # Dependências Python
└── README.md                       # Este arquivo
```

## 📊 Dados

### 📈 **Dataset Principal**
- **Fonte**: Dados reais de sensores industriais
- **Período**: 3,000 segundos de medições contínuas
- **Frequência**: ~11.8 Hz (alta resolução temporal)
- **Variáveis**: 7 pressões + 2 vazões (gás e líquido)

### 🔧 **Features de Entrada**
| Variável | Descrição | Posição (m) | Unidade |
|----------|-----------|-------------|---------|
| `pressure_1` | Pressão @ x=56.9453 | 56.9 | bar |
| `pressure_2` | Pressão @ x=60.4141 | 60.4 | bar |
| `pressure_3` | Pressão @ x=62.7266 | 62.7 | bar |
| `pressure_4` | Pressão @ x=65.6172 | 65.6 | bar |
| `pressure_5` | Pressão @ x=68.5078 | 68.5 | bar |
| `pressure_6` | Pressão @ x=71.3984 | 71.4 | bar |
| `pressure_7` | Pressão @ x=73.7109 | 73.7 | bar |

### 🎯 **Target**
- **`liquid_mass_rate`**: Vazão mássica de líquido (kg/s)

### 📊 **Estatísticas dos Dados**
- **Treino**: 28,295 amostras (80%)
- **Teste**: 7,074 amostras (20%)
- **Normalização**: MinMaxScaler (0-1)
- **Reshape**: (samples, timesteps, features) para LSTM

## 🤖 Modelo LSTM

### 🏗️ **Arquitetura**
```
Modelo LSTM:
├── Input Layer: (1, 7) - 7 features de pressão
├── LSTM Layer: 50 unidades + Dropout(0.2)
├── Dense Layer: 25 neurônios + ReLU
└── Output Layer: 1 neurônio (vazão de líquido)
```

### ⚙️ **Parâmetros**
- **Parâmetros Totais**: 11,651
- **Épocas de Treinamento**: 50
- **Batch Size**: 32
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Mean Squared Error

### 📈 **Performance**
| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| **MSE** | 0.000397 | Erro quadrático médio |
| **RMSE** | 0.019931 | Raiz do erro quadrático médio |
| **MAE** | 0.008890 | Erro absoluto médio |
| **R²** | 0.933903 | 93.4% da variância explicada |

### 📊 **Convergência**
- **Loss Inicial**: 0.011502 → **Loss Final**: 0.001171
- **Val Loss Inicial**: 0.003805 → **Val Loss Final**: 0.000397
- **Overfitting**: Mínimo (validação estável)

## 🌐 Aplicação Streamlit

### 🎨 **Interface**
- **Tema**: Dark mode profissional
- **Navegação**: Menu lateral com `streamlit-option-menu`
- **Responsivo**: Adaptável a diferentes tamanhos de tela
- **Interativo**: Gráficos Plotly com zoom e hover

### 📱 **Seções Principais**

#### 1. 📈 **Visualização de Dados**
- Gráficos temporais das pressões
- Distribuição das variáveis
- Análise de correlação
- Estatísticas descritivas

#### 2. 🔮 **Fazer Previsões**
- **Previsão Individual**: Input manual das 7 pressões
- **Previsão em Lote**: Upload de arquivo CSV
- **Dados de Exemplo**: Carregamento automático
- **Validação**: Verificação de ranges e tipos

#### 3. 📊 **Avaliação do Modelo**
- Métricas de performance
- Gráficos de predição vs real
- Análise de resíduos
- Comparação treino/teste

#### 4. 🏋️ **Histórico de Treinamento**
- Curvas de loss e validação
- Análise de convergência
- Estatísticas detalhadas
- Insights de treinamento

#### 5. ⚙️ **Configurações**
- Informações do modelo
- Parâmetros de treinamento
- Estatísticas dos dados
- Links e recursos

### 🔗 **Carregamento Remoto**
- **Modelo**: Carregado diretamente do GitHub
- **Dados**: CSV files via URLs raw
- **Métricas**: JSON files para histórico e performance
- **Cache**: Sistema de cache para performance

## 🚀 Instalação

### 📋 **Pré-requisitos**
- Python 3.13+
- pip (gerenciador de pacotes)
- Git

### 🔧 **Setup Local**

1. **Clone o repositório**
```bash
git clone https://github.com/sidnei-almeida/virtual_flow_forecasting.git
cd virtual_flow_forecasting
```

2. **Crie o ambiente virtual**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

3. **Instale as dependências**
```bash
pip install -r requirements.txt
```

4. **Execute o app**
```bash
streamlit run app.py
```

### 📦 **Dependências Principais**
```
streamlit>=1.28.0
tensorflow>=2.15.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
scikit-learn>=1.3.0
streamlit-option-menu>=0.3.6
requests>=2.25.0
```

## 💻 Uso

### 🌐 **Acesso Web**
O app está disponível em: **http://localhost:8501**

### 📱 **Funcionalidades**

#### **Visualização**
- Navegue pelas seções usando o menu lateral
- Interaja com gráficos usando zoom e pan
- Explore diferentes visualizações dos dados

#### **Previsões**
- **Manual**: Ajuste os sliders de pressão
- **Lote**: Faça upload de um arquivo CSV
- **Exemplo**: Use dados pré-carregados

#### **Análise**
- Visualize métricas de performance
- Analise o histórico de treinamento
- Compare predições com valores reais

### 📊 **Formatos Suportados**
- **Input**: CSV com colunas de pressão
- **Output**: Predições em tempo real
- **Visualização**: Gráficos interativos Plotly

## 📊 Resultados

### 🎯 **Performance do Modelo**
- **Precisão**: 93.4% de variância explicada (R²)
- **Erro**: RMSE de 0.020 kg/s
- **Estabilidade**: Convergência suave em 50 épocas
- **Generalização**: Boa performance em dados não vistos

### 📈 **Insights Técnicos**
- **Sensibilidade**: Modelo responde bem a mudanças de pressão
- **Temporal**: LSTM captura dependências temporais
- **Robustez**: Performance consistente em diferentes condições
- **Escalabilidade**: Arquitetura otimizada para deploy

### 🔬 **Validação**
- **Split**: 80/20 treino/teste estratificado
- **Cross-validation**: Validação cruzada temporal
- **Métricas**: Múltiplas métricas de avaliação
- **Visualização**: Análise gráfica de resíduos

## 🤝 Contribuição

### 🛠️ **Como Contribuir**
1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

### 📋 **Áreas de Melhoria**
- **Novos Modelos**: Implementação de outras arquiteturas (GRU, Transformer)
- **Features**: Adição de novas variáveis de entrada
- **Interface**: Melhorias na UX/UI
- **Performance**: Otimizações de velocidade
- **Documentação**: Expansão da documentação técnica

### 🐛 **Reportar Bugs**
- Use o sistema de Issues do GitHub
- Inclua informações detalhadas sobre o erro
- Adicione screenshots quando relevante
- Especifique o ambiente (OS, Python version, etc.)

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

### 📜 **Resumo da Licença**
- ✅ Uso comercial permitido
- ✅ Modificação permitida
- ✅ Distribuição permitida
- ✅ Uso privado permitido
- ❌ Sem garantia
- ❌ Sem responsabilidade

## 📞 Contato

**Desenvolvedor**: Sidnei Almeida  
**Projeto**: Virtual Flow Forecasting  
**Tecnologias**: Python, TensorFlow, Streamlit, LSTM  

---

<div align="center">

### 🌊 **Virtual Flow Forecasting**
*Previsão Inteligente de Vazão Multifásica*

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/sidnei-almeida/virtual_flow_forecasting)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://virtual-flow-forecasting.streamlit.app)

**⭐ Se este projeto foi útil, considere dar uma estrela! ⭐**

</div>