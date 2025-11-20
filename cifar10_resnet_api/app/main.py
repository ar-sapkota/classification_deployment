# app/main.py
from fastapi import FastAPI, File, UploadFile
from app.utils import read_image
from app.predict import predict
from app.schemas import PredictionResponse
from app.models import ResNet18CIFAR10
from fastapi.middleware.cors import CORSMiddleware
from app.benchmark import FastAPIBenchmark


app = FastAPI(title="CIFAR-10 Resnet18")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # You can set your Streamlit URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once on startup
model_instance = ResNet18CIFAR10()
model = model_instance.model
class_names = model_instance.class_names


benchmark = FastAPIBenchmark(
    base_url="http://localhost:8001",
    images_folder="benchmark_images"
)


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Receives an image file and returns CIFAR-10 class prediction.
    """
    # 1. Convert uploaded file to PIL Image (async)
    image = await read_image(file)
    
    # 2. Run prediction
    result = predict(image, model, class_names)
    
    # 3. Return structured JSON
    return {
        "class_name": result["class"],
        "confidence": result["confidence"]
    }

@app.get("/status")
async def status():
    """Check if model is loaded"""
    return {
        "model_loaded": model is not None,
        "model_type": "ResNet18",
        "num_classes": len(class_names)
    }

@app.get("/benchmark-report")
def benchmark_report():
    """
    Runs benchmark tests and returns JSON results.
    """
    result = benchmark.run_all()
    return {"benchmark_result": result}