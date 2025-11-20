# app/utils.py
from io import BytesIO
from PIL import Image
from fastapi import UploadFile

async def read_image(file: UploadFile) -> Image.Image:
    """
    Reads the uploaded file asynchronously and returns a PIL Image in RGB format.
    """
    contents = await file.read()  # raw bytes of the uploaded file
    image = Image.open(BytesIO(contents)).convert("RGB")  # convert to RGB
    
    return image
