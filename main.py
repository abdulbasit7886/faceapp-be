from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import gdown
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import io
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    try:
        return int(value) if value is not None else default
    except (ValueError, TypeError):
        return default

def get_env_str(key: str, default: str) -> str:
    return os.getenv(key, default)

# Get configuration from environment variables
HOST = get_env_str("HOST", "0.0.0.0")
PORT = get_env_int("PORT", 8000)
DATASET_URL = get_env_str("DATASET_URL", "")
DATASET_PATH = Path(get_env_str("DATASET_PATH", "dataset.zip"))
UPLOAD_DIR = Path(get_env_str("UPLOAD_DIR", "uploads"))
EMBEDDINGS_DIR = Path(get_env_str("EMBEDDINGS_DIR", "embeddings"))
ALLOWED_ORIGINS = get_env_str("ALLOWED_ORIGINS", "*").split(",")
MAX_FILE_SIZE = get_env_int("MAX_FILE_SIZE", 10485760)  # 10MB default

app = FastAPI(
    title=get_env_str("API_TITLE", "Image Processing API"),
    version=get_env_str("API_VERSION", "1.0.0"),
    description=get_env_str("API_DESCRIPTION", "API for image processing and embedding generation")
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for storing images
UPLOAD_DIR.mkdir(exist_ok=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True)

@app.on_event("startup")
async def startup_event():
    if not DATASET_PATH.exists() and DATASET_URL:
        gdown.download(DATASET_URL, str(DATASET_PATH), quiet=False)
        # Extract dataset if needed
        # shutil.unpack_archive(str(DATASET_PATH), "dataset")

@app.post("/upload-query-image")
async def upload_query_image(file: UploadFile = File(...)):
    try:
        # Check file size
        file_size = 0
        chunk_size = 1024
        chunk = await file.read(chunk_size)
        while chunk:
            file_size += len(chunk)
            if file_size > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail="File too large")
            chunk = await file.read(chunk_size)
        await file.seek(0)
        
        # Save the uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the image (you can add your image processing logic here)
        image = Image.open(file_path)
        
        return {
            "message": "Image uploaded successfully",
            "filename": file.filename,
            "size": f"{os.path.getsize(file_path)} bytes"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-embeddings")
async def generate_embeddings(file: UploadFile = File(...)):
    try:
        # Check file size
        file_size = 0
        chunk_size = 1024
        chunk = await file.read(chunk_size)
        while chunk:
            file_size += len(chunk)
            if file_size > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail="File too large")
            chunk = await file.read(chunk_size)
        await file.seek(0)
        
        # Save the uploaded file
        file_path = EMBEDDINGS_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Generate embeddings (you can add your embedding generation logic here)
        image = Image.open(file_path)
        # Convert image to numpy array for processing
        image_array = np.array(image)
        
        # Placeholder for embedding generation
        # Replace this with your actual embedding generation code
        embedding = np.random.rand(512)  # Example 512-dimensional embedding
        
        return {
            "message": "Embeddings generated successfully",
            "filename": file.filename,
            "embedding_shape": embedding.shape
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Image Processing API is running",
        "version": app.version,
        "title": app.title
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
