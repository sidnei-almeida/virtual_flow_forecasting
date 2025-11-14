"""
Module to load the trained LSTM model
"""

import os
import logging
from pathlib import Path
from tensorflow.keras.models import load_model
import numpy as np

logger = logging.getLogger(__name__)

# Path to the model
MODEL_PATH = Path(__file__).parent / "model" / "meu_modelo_lstm.keras"

class ModelLoader:
    """Class to load and use the LSTM model"""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        """Singleton pattern to ensure the model is loaded only once"""
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def load_model(self):
        """Loads the LSTM model from disk"""
        if self._model is None:
            try:
                if not MODEL_PATH.exists():
                    raise FileNotFoundError(
                        f"Model not found at {MODEL_PATH}. "
                        "Make sure the meu_modelo_lstm.keras file exists."
                    )
                
                logger.info(f"Loading model from {MODEL_PATH}...")
                self._model = load_model(MODEL_PATH)
                logger.info(f"✅ Model loaded successfully! Parameters: {self._model.count_params():,}")
                
            except Exception as e:
                logger.error(f"❌ Error loading model: {e}")
                raise
        
        return self._model
    
    def predict(self, pressures: list) -> float:
        """
        Makes a prediction using the LSTM model
        
        Args:
            pressures: List with 7 pressure values (pressure_1 to pressure_7)
                      Values must be in range [0, 1] (normalized)
        
        Returns:
            float: Predicted liquid flow rate value (normalized)
        
        Raises:
            ValueError: If the number of pressures is not 7 or if any value is out of range
        """
        if len(pressures) != 7:
            raise ValueError(f"Expected 7 pressure values, got {len(pressures)}")
        
        # Validate value range
        pressures_array = np.array(pressures, dtype=np.float32)
        if np.any(pressures_array < 0) or np.any(pressures_array > 1):
            raise ValueError("Pressure values must be in range [0, 1] (normalized)")
        
        # Ensure model is loaded
        if self._model is None:
            self.load_model()
        
        # Reshape for LSTM format: (samples, timesteps, features)
        input_data = np.reshape(pressures_array, (1, 1, 7))
        
        # Make prediction
        prediction = self._model.predict(input_data, verbose=0)
        
        return float(prediction[0][0])
    
    def predict_batch(self, pressures_list: list) -> list:
        """
        Makes multiple batch predictions
        
        Args:
            pressures_list: List of lists, each with 7 pressure values
        
        Returns:
            list: List of predicted values
        """
        if not pressures_list:
            return []
        
        # Validate all inputs
        for i, pressures in enumerate(pressures_list):
            if len(pressures) != 7:
                raise ValueError(f"Sample {i}: expected 7 pressure values, got {len(pressures)}")
        
        # Ensure model is loaded
        if self._model is None:
            self.load_model()
        
        # Prepare batch data
        input_data = np.array(pressures_list, dtype=np.float32)
        input_data_reshaped = np.reshape(input_data, (input_data.shape[0], 1, 7))
        
        # Make predictions
        predictions = self._model.predict(input_data_reshaped, verbose=0)
        
        return predictions.flatten().tolist()


# Global loader instance
model_loader = ModelLoader()

