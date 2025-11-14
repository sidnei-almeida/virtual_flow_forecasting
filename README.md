# 🌊 Virtual Flow Forecasting API

REST API for liquid flow rate prediction using LSTM Deep Learning model.

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Endpoints](#-endpoints)
- [Installation](#-installation)
- [Local Usage](#-local-usage)
- [Deploy on Render](#-deploy-on-render)
- [Usage Examples](#-usage-examples)
- [API Documentation](#-api-documentation)

## 🎯 Overview

This API provides REST endpoints for predicting liquid flow rate in industrial pipe systems using a trained LSTM (Long Short-Term Memory) model. The model uses 7 normalized pressure features (0-1) to predict liquid flow rate.

### 🎯 Objectives

- **RESTful API**: Simple and standardized interface for predictions
- **High Performance**: Single model loading on initialization
- **Easy Deploy**: Ready-to-deploy configuration for Render
- **Auto Documentation**: Integrated Swagger UI

## ✨ Features

- ✅ **Single Prediction**: Endpoint for a single prediction
- ✅ **Batch Prediction**: Endpoint for multiple simultaneous predictions
- ✅ **Health Check**: Monitor API and model status
- ✅ **Model Information**: Endpoint to query architecture and parameters
- ✅ **Metrics**: Access to model evaluation metrics
- ✅ **Data Validation**: Automatic input validation with Pydantic
- ✅ **CORS Enabled**: Ready for frontend integration
- ✅ **Interactive Documentation**: Swagger UI and ReDoc

## 🚀 Endpoints

### GET `/`
Returns basic API information.

**Response:**
```json
{
  "message": "Virtual Flow Forecasting API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

### GET `/health`
Checks API status and if the model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "message": "API and model working correctly"
}
```

### POST `/predict`
Makes a single liquid flow rate prediction.

**Request Body:**
```json
{
  "pressure_1": 0.7761,
  "pressure_2": 0.7281,
  "pressure_3": 0.7361,
  "pressure_4": 0.7560,
  "pressure_5": 0.7811,
  "pressure_6": 0.7690,
  "pressure_7": 0.1330
}
```

**Response:**
```json
{
  "predicted_flow_rate": 0.325634,
  "input_pressures": [0.7761, 0.7281, 0.7361, 0.7560, 0.7811, 0.7690, 0.1330]
}
```

**Validation:**
- All pressure values must be in range [0.0, 1.0]
- All 7 values are required

### POST `/predict/batch`
Makes multiple batch predictions.

**Request Body:**
```json
{
  "pressures": [
    [0.7761, 0.7281, 0.7361, 0.7560, 0.7811, 0.7690, 0.1330],
    [0.7672, 0.7715, 0.7730, 0.7897, 0.8148, 0.8199, 0.4696]
  ]
}
```

**Response:**
```json
{
  "predictions": [0.325634, 0.434126],
  "count": 2
}
```

### GET `/model/info`
Returns information about the LSTM model.

**Response:**
```json
{
  "architecture": "LSTM(50) + Dense(1)",
  "parameters": 11651,
  "input_shape": "(1, 1, 7)",
  "output_shape": "(1,)",
  "features": [
    "pressure_1", "pressure_2", "pressure_3",
    "pressure_4", "pressure_5", "pressure_6", "pressure_7"
  ]
}
```

### GET `/model/metrics`
Returns the model evaluation metrics.

**Response:**
```json
{
  "mse": 0.000397,
  "rmse": 0.019931,
  "mae": 0.008890,
  "r2": 0.933903
}
```

## 📦 Installation

### Prerequisites
- Python 3.11+
- pip

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/sidnei-almeida/virtual_flow_forecasting.git
cd virtual_flow_forecasting
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify the model exists**
```bash
ls model/meu_modelo_lstm.keras
```

## 💻 Local Usage

### Run the server

```bash
uvicorn main:app --reload
```

The API will be available at:
- **API**: http://localhost:8000
- **Swagger Documentation**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc

### Run with custom settings

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 🚀 Deploy on Render

### Automatic Configuration

The project is already configured with `render.yaml`. Follow these steps:

1. **Create a Render account** (if you don't have one)
   - Visit: https://render.com

2. **Connect the repository**
   - In the Render dashboard, click "New +"
   - Select "Blueprint"
   - Connect your GitHub repository

3. **Render automatically detects `render.yaml`**
   - Render will read the `render.yaml` file and configure the service automatically

4. **Wait for deployment**
   - Render will build and deploy automatically
   - The API URL will be provided after deployment

### Manual Configuration (Alternative)

If you prefer to configure manually:

1. **Create a new Web Service**
   - In the dashboard, click "New +" → "Web Service"

2. **Connect the repository**

3. **Configure the service:**
   - **Name**: `virtual-flow-forecasting-api`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

4. **Wait for deployment**

### Environment Variables

No environment variables are required for basic operation. The model will be loaded from the local file.

## 📝 Usage Examples

### Python (requests)

```python
import requests

# API URL (adjust to your Render URL or localhost)
API_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{API_URL}/health")
print(response.json())

# Single prediction
prediction_data = {
    "pressure_1": 0.7761,
    "pressure_2": 0.7281,
    "pressure_3": 0.7361,
    "pressure_4": 0.7560,
    "pressure_5": 0.7811,
    "pressure_6": 0.7690,
    "pressure_7": 0.1330
}

response = requests.post(f"{API_URL}/predict", json=prediction_data)
result = response.json()
print(f"Predicted flow rate: {result['predicted_flow_rate']}")

# Batch prediction
batch_data = {
    "pressures": [
        [0.7761, 0.7281, 0.7361, 0.7560, 0.7811, 0.7690, 0.1330],
        [0.7672, 0.7715, 0.7730, 0.7897, 0.8148, 0.8199, 0.4696],
        [0.7668, 0.7795, 0.7914, 0.8187, 0.8540, 0.8850, 0.6661]
    ]
}

response = requests.post(f"{API_URL}/predict/batch", json=batch_data)
results = response.json()
print(f"Predictions: {results['predictions']}")
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pressure_1": 0.7761,
    "pressure_2": 0.7281,
    "pressure_3": 0.7361,
    "pressure_4": 0.7560,
    "pressure_5": 0.7811,
    "pressure_6": 0.7690,
    "pressure_7": 0.1330
  }'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "pressures": [
      [0.7761, 0.7281, 0.7361, 0.7560, 0.7811, 0.7690, 0.1330],
      [0.7672, 0.7715, 0.7730, 0.7897, 0.8148, 0.8199, 0.4696]
    ]
  }'

# Model information
curl http://localhost:8000/model/info

# Model metrics
curl http://localhost:8000/model/metrics
```

### JavaScript (fetch)

```javascript
// Single prediction
const predictionData = {
  pressure_1: 0.7761,
  pressure_2: 0.7281,
  pressure_3: 0.7361,
  pressure_4: 0.7560,
  pressure_5: 0.7811,
  pressure_6: 0.7690,
  pressure_7: 0.1330
};

fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(predictionData)
})
  .then(response => response.json())
  .then(data => {
    console.log('Predicted flow rate:', data.predicted_flow_rate);
  })
  .catch(error => {
    console.error('Error:', error);
  });
```

## 📚 API Documentation

The API has interactive documentation available at:

- **Swagger UI**: `http://localhost:8000/docs`
  - Interactive interface where you can test all endpoints
  - Shows schemas, examples and allows direct requests

- **ReDoc**: `http://localhost:8000/redoc`
  - Alternative documentation with clean and organized visualization

## 🤖 LSTM Model

### Architecture
- **Input**: 7 normalized pressure features (0-1)
- **LSTM Layer**: 50 units
- **Dense Layer**: 1 neuron (output)
- **Output**: Predicted liquid flow rate (normalized)

### Performance
- **R² Score**: 0.934 (93.4% variance explained)
- **RMSE**: 0.020
- **MAE**: 0.009

### Data Format
- **Input**: Normalized values between 0 and 1
- **Output**: Normalized value between 0 and 1
- To denormalize values, you need to use the MinMaxScaler parameters used in training

## ⚠️ Important Notes

1. **Normalization**: All pressure values must be normalized in range [0, 1]. If you have raw values, you need to normalize them before sending to the API.

2. **Local Model**: The model is loaded from the `model/meu_modelo_lstm.keras` file. Make sure this file exists before deploying.

3. **Performance**: The model is loaded once on application initialization. Subsequent predictions are fast.

4. **CORS**: The API has CORS enabled for all domains. In production, consider restricting to allowed domains.

## 🐛 Troubleshooting

### Error: "Model not found"
- Check if the `model/meu_modelo_lstm.keras` file exists
- Make sure the path is correct

### Error: "Pressure values must be in range [0, 1]"
- Normalize your data before sending to the API
- Values must be between 0.0 and 1.0

### Render Error: "Build failed"
- Check the logs in the Render dashboard
- Make sure all dependencies are in `requirements.txt`
- Verify Python 3.11 is available on Render

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**Developer**: Sidnei Almeida  
**Project**: Virtual Flow Forecasting API  
**Technologies**: Python, FastAPI, TensorFlow, LSTM

---

<div align="center">

### 🌊 **Virtual Flow Forecasting API**
*Intelligent Multiphase Flow Rate Prediction*

⭐ If this project was useful, consider giving it a star! ⭐

</div>
