from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
import shutil
import os
import logging
from contextlib import asynccontextmanager
# Import the new classes from the refactored transcriber module
from .transcriber import AudioProcessingService, TranscriptionConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state for the audio processing service
app_state = {
    "audio_processing_service": None,
    "service_initialization_error": "", # Initialize as empty string
    "config": None # Store config for reference in health check
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup: Initializing AudioProcessingService...")
    try:
        # Create configuration (HF_AUTH_TOKEN should ideally be an env var)
        # For now, ensure it's set in TranscriptionConfig default or as an env var
        config = TranscriptionConfig(
            hf_auth_token=os.getenv("HF_AUTH_TOKEN", "") 
        )
        if not config.hf_auth_token and config.diarization_pipeline_name.startswith("pyannote/"):
             logger.warning("HF_AUTH_TOKEN environment variable is not set. "
                           "Pyannote diarization models may fail to load if they are gated.")
             # You might want to set a placeholder if TranscriptionConfig expects a string
             # config.hf_auth_token = "TOKEN_NOT_SET" # Or handle in TranscriptionConfig

        app_state["config"] = config
        service = AudioProcessingService(config)
        
        # Check if critical components of the service loaded correctly
        if not service.whisper_service or not service.whisper_service.model:
            raise RuntimeError("Whisper service or model failed to initialize within AudioProcessingService.")
        # Pyannote pipeline failure is logged as a warning by AudioProcessingService,
        # allowing transcription to potentially still work.
        
        app_state["audio_processing_service"] = service
        logger.info("AudioProcessingService initialized successfully.")
        logger.info(f"  Whisper model: {config.whisper_model_name} on {service.config.device_str}")
        if service.diarization_service and service.diarization_service.pipeline:
            logger.info(f"  Diarization pipeline: {config.diarization_pipeline_name} loaded.")
        else:
            logger.warning(f"  Diarization pipeline: {config.diarization_pipeline_name} FAILED to load or is unavailable.")
            # Append to the existing error string, ensuring it's a string
            current_error = app_state.get("service_initialization_error", "")
            if not isinstance(current_error, str): # Should not happen if initialized to ""
                current_error = str(current_error) if current_error is not None else ""
            app_state["service_initialization_error"] = current_error + "Diarization pipeline failed to load. "


    except Exception as e:
        logger.error(f"Fatal error during AudioProcessingService initialization: {e}", exc_info=True)
        # Append to the existing error string or set it
        current_error = app_state.get("service_initialization_error", "")
        if not isinstance(current_error, str):
             current_error = str(current_error) if current_error is not None else ""
        app_state["service_initialization_error"] = current_error + str(e)
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
    Returns the status of the AudioProcessingService.
    """
    service: AudioProcessingService = app_state.get("audio_processing_service")
    config: TranscriptionConfig = app_state.get("config")
    init_error = app_state.get("service_initialization_error")

    if init_error:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "AudioProcessingService failed to initialize during startup.",
                "error_details": init_error,
                "whisper_model_configured": config.whisper_model_name if config else "N/A",
                "diarization_pipeline_configured": config.diarization_pipeline_name if config else "N/A",
            }
        )
    
    if service and config:
        whisper_ready = service.whisper_service and service.whisper_service.model is not None
        diarization_ready = service.diarization_service and service.diarization_service.pipeline is not None
        
        status_code = 200
        health_message = "Application is healthy."
        app_status = "ok"

        if not whisper_ready:
            app_status = "error"
            health_message = "Core Whisper service is not available."
            status_code = 503 # Service Unavailable
        elif not diarization_ready:
            app_status = "degraded"
            health_message = "Application is running in a degraded state. Diarization service is not available."
            # status_code remains 200 for degraded, but client should check details

        return JSONResponse(
            status_code=status_code,
            content={
                "status": app_status,
                "message": health_message,
                "device_used": service.config.device_str,
                "whisper_model_status": "loaded" if whisper_ready else "not_loaded",
                "whisper_model_name": config.whisper_model_name,
                "diarization_pipeline_status": "loaded" if diarization_ready else "not_loaded",
                "diarization_pipeline_name": config.diarization_pipeline_name,
            }
        )
    else:
        # Should be caught by init_error, but as a fallback
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": "AudioProcessingService is not available.",
                "details": "Service object not found in application state."
            }
        )

@app.post("/transcribe/", response_class=PlainTextResponse)
async def transcribe_audio_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to transcribe an audio file with speaker diarization.
    Accepts an audio file, saves it temporarily, transcribes and diarizes it,
    and returns plain text with speaker labels.
    """
    service: AudioProcessingService = app_state.get("audio_processing_service")
    init_error = app_state.get("service_initialization_error")

    if init_error or not service:
        logger.error(f"Transcription attempt failed: AudioProcessingService not available. Error: {init_error}")
        raise HTTPException(status_code=503, detail=f"AudioProcessingService not available due to initialization error: {init_error or 'Unknown service error'}")
    
    if not service.whisper_service or not service.whisper_service.model:
        logger.error("Transcription attempt failed: Whisper model component of AudioProcessingService is not loaded.")
        raise HTTPException(status_code=503, detail="Core Whisper model is not available within the processing service.")

    # Diarization pipeline unavailability is handled gracefully by AudioProcessingService,
    # which will log a warning and proceed without speaker labels.

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

        # Use the AudioProcessingService instance to transcribe and diarize
        # This function is expected to return a list of dicts:
        # [{'speaker': 'SPEAKER_00', 'start': 0.0, 'end': 1.5, 'text': 'Hello world'}, ...]
        processed_segments = service.transcribe_and_diarize(temp_file_path)
        
        # Format the output as a single string
        output_lines = []
        current_speaker = None
        current_line = ""
        for segment in processed_segments:
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "")
            if speaker != current_speaker:
                if current_line:
                    output_lines.append(f"{current_speaker}: {current_line.strip()}")
                current_speaker = speaker
                current_line = text
            else:
                current_line += " " + text
        
        if current_line: # Add the last segment
            output_lines.append(f"{current_speaker}: {current_line.strip()}")
            
        final_transcription = "\n".join(output_lines)
        
        logger.info(f"Transcription and diarization successful for '{safe_filename}'.")
        # The transcribe_and_diarize_audio function in transcriber.py now handles file deletion
        return PlainTextResponse(content=final_transcription)

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
