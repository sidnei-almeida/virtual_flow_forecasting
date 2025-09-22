# 🌊 Sistema de Previsão de Vazão de Líquido - App Streamlit

Este é um aplicativo Streamlit completo para visualizar dados, fazer previsões e avaliar um modelo LSTM treinado para previsão de vazão de líquido em sistemas de fluxo multifásico.

## 📋 Funcionalidades

### 🏠 Visão Geral
- Estatísticas gerais do projeto
- Informações sobre o dataset e modelo
- Métricas de performance

### 📊 Visualização dos Dados
- Gráficos temporais das pressões
- Evolução da vazão de líquido
- Matriz de correlação
- Distribuições das variáveis
- Suporte para dados originais, treino e teste

### 🔮 Fazer Previsões
- Interface interativa para entrada de dados
- Previsões individuais com valores customizados
- Opção de usar dados de exemplo do conjunto de teste
- Previsões em lote com visualização
- Métricas de erro em tempo real

### 📈 Avaliação do Modelo
- Métricas completas de performance (MSE, RMSE, MAE, R²)
- Gráfico de dispersão Real vs Previsto
- Comparação temporal
- Análise de resíduos
- Histograma dos resíduos

### ⚙️ Configurações
- Informações técnicas do modelo
- Detalhes do dataset
- Download de dados
- Performance final

## 🚀 Como Executar

### 1. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 2. Executar o App
```bash
streamlit run app.py
```

### 3. Acessar no Navegador
O app será aberto automaticamente em `http://localhost:8501`

## 📁 Estrutura de Arquivos Necessária

```
virtual_flow_forecasting/
├── app.py                              # App Streamlit principal
├── requirements.txt                    # Dependências
├── data/
│   ├── train_data_scaled_manual.csv   # Dados de treino
│   ├── test_data_scaled_manual.csv    # Dados de teste
│   └── riser_pq_uni.csv               # Dados originais
├── model/
│   └── meu_modelo_lstm.keras          # Modelo LSTM treinado
└── notebooks/                         # Notebooks de desenvolvimento
```

## 🔧 Características Técnicas

### Modelo LSTM
- **Arquitetura**: LSTM(50) + Dense(1)
- **Parâmetros**: ~1,550 parâmetros treináveis
- **Função de Perda**: Mean Squared Error
- **Otimizador**: Adam
- **Épocas**: 50
- **Batch Size**: 72

### Features de Entrada
- `pressure_1` a `pressure_7`: 7 sensores de pressão
- **Target**: `liquid_flow_rate` (vazão de líquido)

### Performance
- **MSE**: 0.000463
- **RMSE**: 0.021523
- **MAE**: 0.008902
- **R²**: ~0.99 (excelente ajuste)

## 📊 Visualizações Disponíveis

1. **Gráficos Temporais**: Evolução das pressões e vazão ao longo do tempo
2. **Matriz de Correlação**: Relações entre features e target
3. **Distribuições**: Histogramas das variáveis
4. **Previsões**: Comparação visual entre valores reais e previstos
5. **Análise de Resíduos**: Avaliação da qualidade do modelo
6. **Métricas Interativas**: Performance em tempo real

## 🎯 Casos de Uso

1. **Análise Exploratória**: Explorar os dados e entender padrões
2. **Validação do Modelo**: Verificar performance em dados não vistos
3. **Previsões em Tempo Real**: Fazer previsões com novos dados
4. **Monitoramento**: Acompanhar a qualidade das previsões
5. **Apresentações**: Demonstrar resultados de forma interativa

## 🔍 Funcionalidades Avançadas

- **Cache de Dados**: Carregamento otimizado com `@st.cache_data`
- **Cache de Modelo**: Modelo carregado uma única vez com `@st.cache_resource`
- **Interface Responsiva**: Layout adaptável para diferentes telas
- **Visualizações Interativas**: Gráficos Plotly com zoom, pan, hover
- **Download de Dados**: Exportação dos datasets processados
- **Validação de Entrada**: Interface robusta para entrada de dados

## 🛠️ Personalização

O app pode ser facilmente personalizado modificando:
- Cores e temas no `st.set_page_config()`
- Layout das páginas
- Métricas adicionais
- Visualizações customizadas
- Parâmetros do modelo

## 📈 Próximos Passos

Possíveis melhorias futuras:
- Adicionar mais modelos para comparação
- Implementar validação cruzada
- Adicionar análise de importância das features
- Incluir previsões com intervalos de confiança
- Implementar retreinamento online
- Adicionar logs de uso e monitoramento

## 🐛 Solução de Problemas

### Erro de Carregamento de Dados
Verifique se todos os arquivos estão no local correto:
- `data/train_data_scaled_manual.csv`
- `data/test_data_scaled_manual.csv`
- `data/riser_pq_uni.csv`
- `model/meu_modelo_lstm.keras`

### Erro de Dependências
Execute:
```bash
pip install --upgrade -r requirements.txt
```

### Problemas de Performance
- Use dados menores para visualizações
- Ajuste o tamanho das janelas temporais
- Considere usar cache para operações pesadas
