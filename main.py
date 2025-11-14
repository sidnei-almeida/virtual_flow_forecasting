"""
REST API for liquid flow rate prediction using LSTM model
"""

import os
import logging

# Configure TensorFlow environment BEFORE any imports
# This ensures TensorFlow uses CPU only from the start
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import json
from pathlib import Path
from model_loader import model_loader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Virtual Flow Forecasting API",
    description="REST API for liquid flow rate prediction using LSTM model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS - Allow all origins for development and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Must be False when allow_origins=["*"]
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],  # Allow all common methods
    allow_headers=["*"],  # Allow all headers including Authorization, Content-Type, etc.
    expose_headers=["*"],  # Expose all headers in response
    max_age=3600,  # Cache preflight requests for 1 hour
)


# Pydantic models for data validation
class PressureInput(BaseModel):
    """Input model for a single prediction"""
    pressure_1: float = Field(..., ge=0.0, le=1.0, description="Pressure 1 (normalized 0-1)")
    pressure_2: float = Field(..., ge=0.0, le=1.0, description="Pressure 2 (normalized 0-1)")
    pressure_3: float = Field(..., ge=0.0, le=1.0, description="Pressure 3 (normalized 0-1)")
    pressure_4: float = Field(..., ge=0.0, le=1.0, description="Pressure 4 (normalized 0-1)")
    pressure_5: float = Field(..., ge=0.0, le=1.0, description="Pressure 5 (normalized 0-1)")
    pressure_6: float = Field(..., ge=0.0, le=1.0, description="Pressure 6 (normalized 0-1)")
    pressure_7: float = Field(..., ge=0.0, le=1.0, description="Pressure 7 (normalized 0-1)")
    
    def to_list(self) -> List[float]:
        """Convert to list of pressures"""
        return [
            self.pressure_1, self.pressure_2, self.pressure_3,
            self.pressure_4, self.pressure_5, self.pressure_6, self.pressure_7
        ]


class BatchPressureInput(BaseModel):
    """Input model for batch predictions"""
    pressures: List[List[float]] = Field(
        ...,
        min_items=1,
        description="List of lists, each with 7 normalized pressure values (0-1)"
    )
    
    @validator('pressures')
    def validate_pressures(cls, v):
        """Validate that each list has exactly 7 values in range [0, 1]"""
        for i, pressure_list in enumerate(v):
            if len(pressure_list) != 7:
                raise ValueError(f"Sample {i}: expected 7 pressure values, got {len(pressure_list)}")
            for j, pressure in enumerate(pressure_list):
                if not 0.0 <= pressure <= 1.0:
                    raise ValueError(
                        f"Sample {i}, pressure {j+1}: value {pressure} out of range [0, 1]"
                    )
        return v


class PredictionResponse(BaseModel):
    """Response model for a single prediction"""
    predicted_flow_rate: float = Field(..., description="Predicted liquid flow rate (normalized)")
    input_pressures: List[float] = Field(..., description="Input pressures")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[float] = Field(..., description="List of predicted flow rates (normalized)")
    count: int = Field(..., description="Number of predictions made")


class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    architecture: str
    parameters: int
    input_shape: str
    output_shape: str
    features: List[str]


class MetricsResponse(BaseModel):
    """Response model for model metrics"""
    mse: float
    rmse: float
    mae: float
    r2: float


# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Loads the LSTM model when the application starts"""
    try:
        logger.info("Starting application...")
        model_loader.load_model()
        logger.info("✅ Model loaded successfully on startup!")
    except Exception as e:
        logger.error(f"❌ Error loading model on startup: {e}")
        raise


# Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - Returns basic API information"""
    return {
        "message": "Virtual Flow Forecasting API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "base_url": "https://virtual-flow-forecasting.onrender.com"
    }


@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    """Handle OPTIONS requests for CORS preflight - Required for CORS to work properly"""
    from fastapi.responses import Response
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "3600"
        }
    )


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        # Check if model is loaded
        model = model_loader.load_model()
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "message": "API and model working correctly"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "error": str(e)
        }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(pressure_input: PressureInput):
    """
    Makes a single liquid flow rate prediction
    
    Receives 7 normalized pressure values (0-1) and returns the predicted flow rate.
    """
    try:
        pressures = pressure_input.to_list()
        predicted_flow = model_loader.predict(pressures)
        
        return PredictionResponse(
            predicted_flow_rate=predicted_flow,
            input_pressures=pressures
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(batch_input: BatchPressureInput):
    """
    Makes multiple batch predictions
    
    Receives a list of lists, each with 7 normalized pressure values (0-1).
    Returns a list with all predicted flow rates.
    """
    try:
        predictions = model_loader.predict_batch(batch_input.pressures)
        
        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error making batch predictions: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Returns information about the LSTM model"""
    try:
        model = model_loader.load_model()
        
        return ModelInfoResponse(
            architecture="LSTM(50) + Dense(1)",
            parameters=model.count_params(),
            input_shape="(1, 1, 7)",
            output_shape="(1,)",
            features=[
                "pressure_1", "pressure_2", "pressure_3",
                "pressure_4", "pressure_5", "pressure_6", "pressure_7"
            ]
        )
    except Exception as e:
        logger.error(f"Error getting model information: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/model/metrics", response_model=MetricsResponse, tags=["Model"])
async def model_metrics():
    """Returns the model evaluation metrics"""
    try:
        metrics_path = Path(__file__).parent / "model" / "model_metrics.json"
        
        if not metrics_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Metrics file not found"
            )
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        return MetricsResponse(
            mse=metrics.get("mse", 0.0),
            rmse=metrics.get("rmse", 0.0),
            mae=metrics.get("mae", 0.0),
            r2=metrics.get("r2", 0.0)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # For local development
    uvicorn.run(app, host="0.0.0.0", port=8000)

