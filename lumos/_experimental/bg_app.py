from fastapi import FastAPI, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse
import os
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Directory to save uploaded and processed files
UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
logger.info(f"Created directories: {UPLOAD_DIR} and {PROCESSED_DIR}")


def process_file(file_path: str, output_path: str):
    logger.info(f"Starting processing of file: {file_path}")
    # Simulate a time-consuming processing task
    time.sleep(20)  # Replace with actual processing logic
    # For demonstration, we'll just copy the file to the processed directory
    try:
        with open(file_path, "rb") as src, open(output_path, "wb") as dst:
            dst.write(src.read())
        logger.info(f"Successfully processed file: {file_path} -> {output_path}")
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        raise


@app.post("/upload/")
async def upload_file(file: UploadFile, background_tasks: BackgroundTasks):
    logger.info(f"Received upload request for file: {file.filename}")
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_location, "wb") as f:
            f.write(await file.read())
        logger.info(f"Successfully saved uploaded file to: {file_location}")
        
        output_location = os.path.join(PROCESSED_DIR, file.filename)
        background_tasks.add_task(process_file, file_location, output_location)
        logger.info(f"Added processing task for file: {file.filename}")
        
        return {"message": "File uploaded successfully. Processing in background.", "filename": file.filename}
    except Exception as e:
        logger.error(f"Error handling upload for {file.filename}: {str(e)}")
        raise


@app.get("/download/{filename}")
async def download_file(filename: str):
    logger.info(f"Received download request for file: {filename}")
    file_path = os.path.join(PROCESSED_DIR, filename)
    if os.path.exists(file_path):
        logger.info(f"Serving file: {file_path}")
        return FileResponse(file_path, media_type='application/octet-stream', filename=filename)
    else:
        logger.warning(f"File not found: {file_path}")
        return {"message": "File not found or still processing."}
