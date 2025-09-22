# 🚀 Como Usar o App Streamlit - Previsão de Vazão de Líquido

## 📋 Resumo do Projeto

Este app Streamlit fornece uma interface completa para visualizar dados, fazer previsões e avaliar um modelo LSTM treinado para previsão de vazão de líquido em sistemas de fluxo multifásico.

## 🎯 Funcionalidades Principais

### 🏠 **Visão Geral**
- Estatísticas do dataset (35.369 amostras)
- Informações do modelo LSTM (11.651 parâmetros)
- Métricas de performance

### 📊 **Visualização dos Dados**
- Gráficos temporais das pressões
- Evolução da vazão de líquido
- Matriz de correlação interativa
- Distribuições das variáveis
- Suporte para dados originais, treino e teste

### 🔮 **Fazer Previsões**
- Interface para entrada manual de dados
- Previsões com dados de exemplo
- Previsões em lote com visualização
- Métricas de erro em tempo real

### 📈 **Avaliação do Modelo**
- Métricas completas (MSE, RMSE, MAE, R²)
- Gráfico de dispersão Real vs Previsto
- Análise de resíduos
- Comparação temporal

### 🏋️ **Histórico de Treinamento** (NOVO!)
- Curvas de perda durante o treinamento
- Análise de convergência
- Detecção de overfitting
- Estatísticas detalhadas do treinamento
- Visualização logarítmica das perdas

### ⚙️ **Configurações**
- Informações técnicas detalhadas
- Download de datasets
- Performance final do modelo

## 🚀 Execução Rápida

### Opção 1: Script Automático
```bash
./run_app.sh
```

### Opção 2: Manual
```bash
# Ativar ambiente virtual
source venv/bin/activate

# Executar o app
streamlit run app.py
```

### Opção 3: Testar Primeiro
```bash
# Executar testes
python test_app.py

# Se todos passarem, executar o app
streamlit run app.py
```

## 📱 Acesso ao App

Após executar, o app estará disponível em:
- **URL Local**: http://localhost:8501
- **URL da Rede**: http://[seu-ip]:8501

## 🎮 Guia de Uso

### 1. **Explorar os Dados**
1. Vá para "📊 Visualização dos Dados"
2. Selecione o tipo de dados (Originais, Treino, Teste)
3. Escolha as pressões para visualizar
4. Explore as correlações e distribuições

### 2. **Fazer Previsões**
1. Acesse "🔮 Fazer Previsões"
2. Insira valores das pressões (0-1, valores escalonados)
3. Ou use "Usar dados de exemplo" para testar
4. Clique em "🔮 Fazer Previsão"

### 3. **Avaliar o Modelo**
1. Vá para "📈 Avaliação do Modelo"
2. Veja as métricas de performance
3. Analise os gráficos de comparação
4. Examine a análise de resíduos

### 4. **Analisar o Treinamento** (NOVO!)
1. Acesse "🏋️ Histórico de Treinamento"
2. Veja as curvas de perda durante o treinamento
3. Analise a convergência do modelo
4. Verifique se há overfitting
5. Examine estatísticas detalhadas

### 5. **Configurações**
1. Acesse "⚙️ Configurações"
2. Veja informações técnicas do modelo
3. Baixe os datasets se necessário

## 📊 Exemplo de Uso

### Previsão Individual
1. **Entrada**: Valores das 7 pressões (ex: 0.5, 0.6, 0.4, 0.7, 0.3, 0.8, 0.2)
2. **Resultado**: Vazão de líquido prevista (ex: 0.515841)
3. **Interpretação**: Valor escalonado entre 0-1

### Previsões em Lote
1. Selecione número de amostras (ex: 100)
2. Clique em "Gerar Previsões em Lote"
3. Veja comparação visual e métricas

## 🔧 Características Técnicas

### Modelo LSTM
- **Arquitetura**: LSTM(50) + Dense(1)
- **Parâmetros**: 11.651
- **Performance**: MSE=0.000463, RMSE=0.021523, MAE=0.008902
- **R²**: ~0.99 (excelente ajuste)

### Dados
- **Features**: 7 pressões escalonadas (0-1)
- **Target**: Vazão de líquido escalonada
- **Divisão**: 80% treino / 20% teste
- **Temporal**: Dados de série temporal

## 🎨 Interface

### Navegação
- **Sidebar**: Menu principal com 5 seções
- **Layout**: Responsivo, adapta-se ao tamanho da tela
- **Tema**: Azul e branco, limpo e profissional

### Visualizações
- **Plotly**: Gráficos interativos com zoom, pan, hover
- **Streamlit**: Componentes nativos (sliders, selectboxes, etc.)
- **Responsivo**: Adapta-se a diferentes dispositivos

## 🛠️ Personalização

### Modificar Cores
Edite `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
```

### Adicionar Métricas
Modifique `app.py` na seção de avaliação.

### Novos Gráficos
Use Plotly para criar visualizações customizadas.

## 🐛 Solução de Problemas

### App não inicia
```bash
# Verificar dependências
python test_app.py

# Reinstalar dependências
pip install -r requirements.txt
```

### Erro de dados
Verifique se os arquivos estão em:
- `data/train_data_scaled_manual.csv`
- `data/test_data_scaled_manual.csv`
- `data/riser_pq_uni.csv`
- `model/meu_modelo_lstm.keras`

### Performance lenta
- Use janelas menores para visualizações
- Reduza o número de amostras para previsões em lote
- O modelo usa cache para otimização

## 📈 Próximos Passos

### Melhorias Sugeridas
1. **Novos Modelos**: Adicionar outros algoritmos para comparação
2. **Validação Cruzada**: Implementar validação temporal
3. **Intervalos de Confiança**: Adicionar incerteza nas previsões
4. **Retreinamento**: Interface para atualizar o modelo
5. **Logs**: Sistema de monitoramento de uso

### Expansões Possíveis
1. **Análise de Importância**: Mostrar quais pressões são mais importantes
2. **Previsões Multi-step**: Prever múltiplos passos à frente
3. **Alertas**: Sistema de alertas para valores anômalos
4. **API**: Transformar em API REST
5. **Deploy**: Deploy em cloud (Heroku, AWS, etc.)

## 📞 Suporte

Se encontrar problemas:
1. Execute `python test_app.py` para diagnóstico
2. Verifique se todos os arquivos estão presentes
3. Confirme que o ambiente virtual está ativo
4. Verifique as versões das dependências

## 🎉 Conclusão

Este app Streamlit fornece uma interface completa e profissional para trabalhar com o modelo LSTM de previsão de vazão de líquido. Com visualizações interativas, previsões em tempo real e avaliação completa do modelo, é uma ferramenta poderosa para análise e demonstração dos resultados.

**Divirta-se explorando os dados e fazendo previsões! 🌊📊**
