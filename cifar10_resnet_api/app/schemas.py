# app/schemas.py
from pydantic import BaseModel

class PredictionResponse(BaseModel):
    class_name: str        # Name of the predicted class
    confidence: float      #  confidence score