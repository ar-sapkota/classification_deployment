from torchvision import transforms
from PIL import Image
import torch

# Define preprocessing pipeline
transform = transforms.Compose([             # CIFAR-10 expects 32x32 images
    transforms.Resize((32, 32)),
    transforms.ToTensor(),                    # Convert PIL image to PyTorch tensor
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],   # CIFAR-10 mean
        std=[0.247, 0.243, 0.261]
    )  # Normalize RGB to [-1,1]
])

# Prediction function
def predict(image: Image.Image, model, class_names):
    """
    Preprocesses the image and predicts its class using GoogLeNet.
    Args:
        image (PIL.Image): Input image
    Returns:
        dict: {'class': <class_name>, 'confidence': <float>}
    """
    # Preprocess
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Forward pass
    with torch.no_grad():
        outputs = model(input_tensor)          # Model output logits
        probabilities = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
        confidence, predicted_class = torch.max(probabilities, dim=1)  # Highest probability

    # Return results
    return {
        "class": class_names[predicted_class.item()],
        "confidence": float(confidence.item())
   }