#!/bin/bash

# Script para executar o app Streamlit de Previsão de Vazão de Líquido

echo "🌊 Iniciando Sistema de Previsão de Vazão de Líquido..."
echo ""

# Verificar se o ambiente virtual existe
if [ ! -d "venv" ]; then
    echo "❌ Ambiente virtual não encontrado. Criando..."
    python -m venv venv
fi

# Ativar ambiente virtual
echo "🔄 Ativando ambiente virtual..."
source venv/bin/activate

# Instalar dependências se necessário
echo "📦 Verificando dependências..."
pip install -r requirements.txt

# Verificar se os arquivos necessários existem
echo "🔍 Verificando arquivos necessários..."

files=(
    "data/train_data_scaled_manual.csv"
    "data/test_data_scaled_manual.csv" 
    "data/riser_pq_uni.csv"
    "model/meu_modelo_lstm.keras"
)

for file in "${files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Arquivo não encontrado: $file"
        exit 1
    else
        echo "✅ $file"
    fi
done

echo ""
echo "🚀 Iniciando aplicativo Streamlit..."
echo "📱 O app será aberto em: http://localhost:8501"
echo ""

# Executar o app
streamlit run app.py
