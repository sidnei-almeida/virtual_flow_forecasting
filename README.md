<div align="center">

# 🌊 Virtual Flow Forecasting API

**REST API for Intelligent Multiphase Flow Rate Prediction using LSTM Deep Learning**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

*Production-ready REST API for predicting liquid flow rates in industrial pipe systems*

[Features](#-features) • [Quick Start](#-quick-start) • [API Documentation](#-api-documentation) • [Deployment](#-deployment) • [Examples](#-examples)

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [API Documentation](#-api-documentation)
- [Endpoints](#-endpoints)
- [Examples](#-examples)
- [Deployment](#-deployment)
- [Model Information](#-model-information)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

**Virtual Flow Forecasting API** is a production-ready REST API that leverages advanced LSTM (Long Short-Term Memory) neural networks to predict liquid flow rates in industrial pipe systems. The API processes 7 normalized pressure sensor readings and returns accurate flow rate predictions in real-time.

### Key Highlights

- 🚀 **Production-Ready**: Built with FastAPI for high performance and reliability
- 🤖 **Deep Learning**: Powered by TensorFlow/Keras LSTM model (93.4% R² score)
- 📊 **RESTful Design**: Clean, standardized API endpoints following REST principles
- 🔄 **Batch Processing**: Efficient batch prediction endpoint for multiple samples
- 📚 **Auto Documentation**: Interactive Swagger UI and ReDoc included
- 🌐 **CORS Enabled**: Ready for frontend integration out of the box

---

## ✨ Features

### Core Functionality

| Feature | Description |
|---------|-------------|
| **Single Prediction** | Real-time prediction endpoint for individual flow rate forecasts |
| **Batch Prediction** | Efficient bulk processing for multiple samples in one request |
| **Health Monitoring** | Built-in health check endpoint for system status verification |
| **Model Information** | Detailed model architecture and parameter information |
| **Performance Metrics** | Access to evaluation metrics (MSE, RMSE, MAE, R²) |

### Technical Features

- ✅ **Input Validation**: Automatic validation using Pydantic models
- ✅ **Error Handling**: Comprehensive error responses with detailed messages
- ✅ **Model Caching**: Singleton pattern for efficient model loading
- ✅ **Type Safety**: Full type hints and Pydantic schema validation
- ✅ **Async Support**: Built on FastAPI's async framework
- ✅ **Interactive Docs**: Swagger UI and ReDoc for easy testing

---

## 🏗️ Architecture

### System Architecture

```
┌─────────────────┐
│   Frontend      │
│   (Client)      │
└────────┬────────┘
         │ HTTP/REST
         ▼
┌─────────────────┐
│   FastAPI       │
│   (main.py)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Loader   │
│ (model_loader)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LSTM Model    │
│  (TensorFlow)   │
└─────────────────┘
```

### Model Architecture

```
Input Layer:    7 pressure features (normalized 0-1)
        ↓
LSTM Layer:     50 units
        ↓
Dense Layer:    1 unit (output)
        ↓
Output:         Liquid flow rate (normalized 0-1)
```

### Tech Stack

- **Framework**: FastAPI 0.104+
- **ML Framework**: TensorFlow 2.13+
- **Validation**: Pydantic 2.0+
- **Server**: Uvicorn
- **Language**: Python 3.11+

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Installation Steps

<details>
<summary><b>1. Clone the Repository</b></summary>

```bash
git clone https://github.com/sidnei-almeida/virtual_flow_forecasting.git
cd virtual_flow_forecasting
```
</details>

<details>
<summary><b>2. Create Virtual Environment</b></summary>

```bash
# Linux/Mac
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```
</details>

<details>
<summary><b>3. Install Dependencies</b></summary>

```bash
pip install -r requirements.txt
```
</details>

<details>
<summary><b>4. Verify Model File</b></summary>

```bash
ls model/meu_modelo_lstm.keras
```
</details>

<details>
<summary><b>5. Run the API</b></summary>

```bash
uvicorn main:app --reload
```

The API will be available at:
- **API**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
</details>

---

## 📚 API Documentation

### Base URL

```
Production:  [Provided after Render deployment]
Development: http://localhost:8000
```

### Interactive Documentation

- **Swagger UI**: `/docs` - Interactive API explorer
- **ReDoc**: `/redoc` - Alternative documentation format

---

## 🔌 Endpoints

### Root Endpoint

```http
GET /
```

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

---

### Health Check

```http
GET /health
```

Verifies API status and model availability.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "message": "API and model working correctly"
}
```

---

### Single Prediction

```http
POST /predict
```

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
- All 7 pressure fields are **required**
- All values must be floats between **0.0 and 1.0** (inclusive)

---

### Batch Predictions

```http
POST /predict/batch
```

Makes multiple predictions in a single request.

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

---

### Model Information

```http
GET /model/info
```

Returns detailed information about the LSTM model.

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

---

### Model Metrics

```http
GET /model/metrics
```

Returns model evaluation metrics.

**Response:**
```json
{
  "mse": 0.000397,
  "rmse": 0.019931,
  "mae": 0.008890,
  "r2": 0.933903
}
```

---

## 💻 Examples

### Python Example

<details>
<summary><b>Single Prediction</b></summary>

```python
import requests

API_URL = "http://localhost:8000"

# Make a prediction
response = requests.post(
    f"{API_URL}/predict",
    json={
        "pressure_1": 0.7761,
        "pressure_2": 0.7281,
        "pressure_3": 0.7361,
        "pressure_4": 0.7560,
        "pressure_5": 0.7811,
        "pressure_6": 0.7690,
        "pressure_7": 0.1330
    }
)

result = response.json()
print(f"Predicted flow rate: {result['predicted_flow_rate']}")
```
</details>

<details>
<summary><b>Batch Prediction</b></summary>

```python
import requests

API_URL = "http://localhost:8000"

# Make batch predictions
response = requests.post(
    f"{API_URL}/predict/batch",
    json={
        "pressures": [
            [0.7761, 0.7281, 0.7361, 0.7560, 0.7811, 0.7690, 0.1330],
            [0.7672, 0.7715, 0.7730, 0.7897, 0.8148, 0.8199, 0.4696],
            [0.7668, 0.7795, 0.7914, 0.8187, 0.8540, 0.8850, 0.6661]
        ]
    }
)

results = response.json()
print(f"Predictions: {results['predictions']}")
print(f"Count: {results['count']}")
```
</details>

### JavaScript Example

<details>
<summary><b>Using Fetch API</b></summary>

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
</details>

<details>
<summary><b>Using Async/Await</b></summary>

```javascript
async function predictFlowRate(pressures) {
  try {
    const response = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(pressures)
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    return data.predicted_flow_rate;
  } catch (error) {
    console.error('Prediction error:', error);
    throw error;
  }
}

// Usage
const pressures = {
  pressure_1: 0.7761,
  pressure_2: 0.7281,
  pressure_3: 0.7361,
  pressure_4: 0.7560,
  pressure_5: 0.7811,
  pressure_6: 0.7690,
  pressure_7: 0.1330
};

predictFlowRate(pressures)
  .then(flowRate => console.log('Flow rate:', flowRate))
  .catch(error => console.error('Error:', error));
```
</details>

### cURL Examples

<details>
<summary><b>Health Check</b></summary>

```bash
curl http://localhost:8000/health
```
</details>

<details>
<summary><b>Single Prediction</b></summary>

```bash
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
```
</details>

---

## 🚀 Deployment

### Deploy on Render

The project includes a `render.yaml` configuration file for automatic deployment.

#### Automatic Setup

1. **Create Render Account**
   - Visit [render.com](https://render.com)
   - Sign up or log in

2. **Connect Repository**
   - Click "New +" → "Blueprint"
   - Connect your GitHub repository

3. **Automatic Configuration**
   - Render detects `render.yaml` automatically
   - Service configuration is applied

4. **Deploy**
   - Render builds and deploys automatically
   - API URL is provided after deployment

#### Manual Configuration

If you prefer manual setup:

| Setting | Value |
|---------|-------|
| **Name** | `virtual-flow-forecasting-api` |
| **Environment** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn main:app --host 0.0.0.0 --port $PORT` |

---

## 🤖 Model Information

### Architecture Details

| Component | Specification |
|-----------|--------------|
| **Model Type** | LSTM Neural Network |
| **Input Features** | 7 normalized pressure values (0-1) |
| **LSTM Units** | 50 |
| **Output** | 1 (liquid flow rate, normalized) |
| **Total Parameters** | 11,651 |
| **Training Epochs** | 50 |
| **Batch Size** | 72 |
| **Optimizer** | Adam |
| **Loss Function** | Mean Squared Error |

### Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² Score** | 0.934 | 93.4% variance explained |
| **RMSE** | 0.020 | Root mean squared error |
| **MAE** | 0.009 | Mean absolute error |
| **MSE** | 0.000397 | Mean squared error |

### Data Normalization

⚠️ **Important**: All pressure values must be normalized to range **[0.0, 1.0]**

If you have raw pressure values, normalize them using:

```
normalized_value = (raw_value - min_value) / (max_value - min_value)
```

The predicted flow rate is also normalized and may need denormalization using your scaling parameters.

---

## 🔧 Error Handling

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| `200` | OK | Request successful |
| `400` | Bad Request | Validation error (check input format/values) |
| `404` | Not Found | Resource not found |
| `500` | Internal Server Error | Server-side error |

### Error Response Format

```json
{
  "detail": "Error message description"
}
```

### Common Errors

| Error | Status | Solution |
|-------|--------|----------|
| Missing required field | `422` | Include all 7 pressure fields |
| Invalid value range | `400` | Ensure values are between 0.0 and 1.0 |
| Wrong number of values | `400` | Each sample must have exactly 7 values |
| Model not loaded | `500` | Check server logs, verify model file exists |

---

## 📁 Project Structure

```
virtual_flow_forecasting/
├── main.py                 # FastAPI application
├── model_loader.py         # Model loading module
├── requirements.txt        # Python dependencies
├── render.yaml            # Render deployment config
├── runtime.txt            # Python version
├── API_ENDPOINTS.txt      # Frontend documentation
├── README.md              # This file
│
├── model/
│   ├── meu_modelo_lstm.keras      # Trained LSTM model
│   ├── model_metrics.json         # Evaluation metrics
│   └── training_history.json      # Training history
│
├── data/
│   ├── train_data_scaled_manual.csv
│   ├── test_data_scaled_manual.csv
│   └── riser_pq_uni.csv
│
└── notebooks/
    ├── 1. Data Pre-Processing.ipynb
    └── 2. LSTM Model Training.ipynb
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Include docstrings for all public functions
- Write tests for new features
- Update documentation as needed

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### License Summary

- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Private use allowed
- ❌ No warranty provided
- ❌ No liability assumed

---

## 📞 Contact & Support

**Developer**: Sidnei Almeida  
**Project**: Virtual Flow Forecasting API  
**Technologies**: Python, FastAPI, TensorFlow, LSTM

### Resources

- **API Documentation**: `/docs` (Swagger UI) or `/redoc` (ReDoc)
- **Frontend Guide**: See `API_ENDPOINTS.txt`
- **Issues**: Use GitHub Issues for bug reports
- **Questions**: Contact the development team

---

<div align="center">

### ⭐ If this project was useful, consider giving it a star! ⭐

**Built with ❤️ using FastAPI and TensorFlow**

[![GitHub stars](https://img.shields.io/github/stars/sidnei-almeida/virtual_flow_forecasting?style=social)](https://github.com/sidnei-almeida/virtual_flow_forecasting/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/sidnei-almeida/virtual_flow_forecasting?style=social)](https://github.com/sidnei-almeida/virtual_flow_forecasting/network/members)

</div>
