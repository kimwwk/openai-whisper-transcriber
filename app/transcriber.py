import whisper
import torch
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "small.en"
_model = None
_device = None

def load_model():
    """
    Loads the Whisper model.
    Tries to load on CUDA, falls back to CPU if CUDA is not available or fails.
    Caches the model and device globally.
    """
    global _model, _device
    if _model is not None:
        logger.info(f"Model '{MODEL_NAME}' already loaded on device '{_device}'.")
        return _model, _device

    try:
        if torch.cuda.is_available():
            logger.info(f"CUDA is available. Attempting to load model '{MODEL_NAME}' on GPU.")
            _model = whisper.load_model(MODEL_NAME, device="cuda")
            _device = "cuda"
            logger.info(f"Successfully loaded model '{MODEL_NAME}' on CUDA GPU.")
        else:
            raise RuntimeError("CUDA not available")
    except Exception as e_cuda:
        logger.warning(f"Failed to load model '{MODEL_NAME}' on CUDA: {e_cuda}. Falling back to CPU.")
        try:
            _model = whisper.load_model(MODEL_NAME, device="cpu")
            _device = "cpu"
            logger.info(f"Successfully loaded model '{MODEL_NAME}' on CPU.")
        except Exception as e_cpu:
            logger.error(f"Failed to load model '{MODEL_NAME}' on CPU: {e_cpu}")
            _model = None
            _device = None
            raise  # Re-raise the exception if CPU loading also fails

    return _model, _device

def transcribe_audio(audio_path: str):
    """
    Transcribes the audio file at the given path using the pre-loaded model.
    """
    global _model
    if _model is None:
        logger.error("Transcription called but model is not loaded.")
        raise RuntimeError("Model not loaded. Call load_model() first or check startup logs.")

    try:
        logger.info(f"Transcribing audio file: {audio_path} using model on device: {_device}")
        # Ensure the model is in evaluation mode if applicable (good practice, though Whisper handles this)
        # _model.eval() 
        result = _model.transcribe(audio_path, fp16=(_device=="cuda")) # fp16 only for CUDA
        transcription = result["text"]
        logger.info(f"Transcription successful for: {audio_path}")
        return transcription.strip() if transcription else ""
    except Exception as e:
        logger.error(f"Error during transcription for {audio_path}: {e}")
        raise
    finally:
        # Clean up the temporary audio file if it exists
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                logger.info(f"Temporary file {audio_path} removed.")
            except Exception as e_remove:
                logger.error(f"Error removing temporary file {audio_path}: {e_remove}")

if __name__ == '__main__':
    # Example usage (for testing this module directly)
    # This part will not run when imported by FastAPI
    try:
        model, device = load_model()
        if model:
            logger.info(f"Model loaded on {device} for direct testing.")
            # Create a dummy audio file for testing if you have one
            # e.g., with open("dummy_audio.wav", "w") as f: f.write("dummy")
            # transcription = transcribe_audio("dummy_audio.wav")
            # logger.info(f"Test transcription: {transcription}")
        else:
            logger.error("Failed to load model for direct testing.")
    except Exception as e:
        logger.error(f"Error in direct test: {e}")
