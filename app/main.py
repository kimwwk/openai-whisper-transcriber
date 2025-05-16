from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
import shutil
import os
import logging
from contextlib import asynccontextmanager
from . import transcriber as transcriber_module # Use a different name to avoid conflict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state for the model and device
app_state = {
    "model": None,
    "device": None,
    "model_load_error": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model on startup
    logger.info("Application startup: Loading Whisper model...")
    try:
        model, device = transcriber_module.load_model()
        app_state["model"] = model
        app_state["device"] = device
        logger.info(f"Model '{transcriber_module.MODEL_NAME}' loaded successfully on device '{device}'.")
    except Exception as e:
        logger.error(f"Fatal error during model loading on startup: {e}", exc_info=True)
        app_state["model_load_error"] = str(e)
        # Depending on policy, you might want the app to not start or to run in a degraded state.
        # For now, it will run but /health will show the error and /transcribe will fail.
    yield
    # Clean up resources on shutdown (if any)
    logger.info("Application shutdown.")

app = FastAPI(lifespan=lifespan)

# Define a temporary directory for uploads
TEMP_UPLOAD_DIR = "/tmp/transcriber_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

@app.get("/health", response_class=JSONResponse)
async def health_check():
    """
    Health check endpoint.
    Returns the status of the model and the device it's loaded on.
    """
    if app_state["model_load_error"]:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Model failed to load during startup.",
                "model_status": "error_loading",
                "error_details": app_state["model_load_error"],
                "whisper_model": transcriber_module.MODEL_NAME,
            }
        )
    if app_state["model"] and app_state["device"]:
        return {
            "status": "ok",
            "message": "Application is healthy.",
            "model_status": "loaded",
            "device_used": app_state["device"],
            "whisper_model": transcriber_module.MODEL_NAME,
        }
    else:
        # This case should ideally be covered by model_load_error,
        # but as a fallback:
        return JSONResponse(
            status_code=503, # Service Unavailable
            content={
                "status": "error",
                "message": "Model is not available.",
                "model_status": "not_loaded",
                "whisper_model": transcriber_module.MODEL_NAME,
            }
        )

@app.post("/transcribe/", response_class=PlainTextResponse)
async def transcribe_audio_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to transcribe an audio file.
    Accepts an audio file, saves it temporarily, transcribes it, and returns plain text.
    """
    if app_state["model_load_error"] or not app_state["model"]:
        logger.error("Transcription attempt failed: Model is not loaded due to startup error.")
        raise HTTPException(status_code=503, detail=f"Model not available due to loading error: {app_state['model_load_error']}")

    temp_file_path = None
    try:
        # Ensure the filename is safe
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided.")
        
        safe_filename = os.path.basename(file.filename) # Basic sanitization
        temp_file_path = os.path.join(TEMP_UPLOAD_DIR, safe_filename)

        logger.info(f"Receiving file: {safe_filename}")
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File '{safe_filename}' saved temporarily to '{temp_file_path}'.")

        transcription = transcriber_module.transcribe_audio(temp_file_path)
        logger.info(f"Transcription successful for '{safe_filename}'.")
        # The transcribe_audio function in transcriber.py now handles file deletion
        return PlainTextResponse(content=transcription)

    except HTTPException:
        raise # Re-raise HTTPException directly
    except Exception as e:
        logger.error(f"Error processing file {file.filename or 'unknown'}: {e}", exc_info=True)
        # Ensure temporary file is cleaned up even on error, if it wasn't handled by transcribe_audio's finally block
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Temporary file {temp_file_path} cleaned up after error.")
            except Exception as e_remove:
                logger.error(f"Error removing temporary file {temp_file_path} after error: {e_remove}")
        raise HTTPException(status_code=500, detail=f"An error occurred during transcription: {str(e)}")
    finally:
        # Ensure file object is closed
        if hasattr(file, 'file') and hasattr(file.file, 'close'):
            file.file.close()

if __name__ == "__main__":
    # This part is for local development testing (not used by uvicorn in Docker)
    import uvicorn
    logger.info("Starting Uvicorn server for local development...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
