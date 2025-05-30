import torch
import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# External library imports
from pyannote.audio import Pipeline as PyannotePipeline # Renamed to avoid conflict
from pyannote.core import Annotation as PyannoteAnnotation
import whisper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
@dataclass
class TranscriptionConfig:
    """Configuration for the audio processing services."""
    whisper_model_name: str = "small.en"
    diarization_pipeline_name: str = "pyannote/speaker-diarization-3.1"
    # IMPORTANT: For production, HF_AUTH_TOKEN should come from environment variables or secure config
    hf_auth_token: Optional[str] = os.getenv("HF_AUTH_TOKEN", "") # Replace with your token or ensure env var is set
    audio_convert_sample_rate: int = 16000
    audio_convert_channels: int = 1
    # Field for device, will be determined at runtime
    device_str: Optional[str] = field(init=False, default=None)

    def __post_init__(self):
        if self.hf_auth_token == "":
            logger.warning("Using default/placeholder Hugging Face auth token. "
                           "Ensure HF_AUTH_TOKEN environment variable is set or update TranscriptionConfig.")

# --- Device Management Utility ---
class DeviceManager:
    @staticmethod
    def get_device_str() -> str:
        """Determines and returns the device string ('cuda' or 'cpu')."""
        if torch.cuda.is_available():
            logger.info("CUDA is available. Using GPU.")
            return "cuda"
        else:
            logger.info("CUDA not available. Using CPU.")
            return "cpu"

    @staticmethod
    def get_torch_device() -> torch.device:
        """Determines and returns the torch.device object."""
        return torch.device(DeviceManager.get_device_str())

# --- Audio Conversion Utility ---
class AudioConverter:
    @staticmethod
    def convert_to_wav(input_path: str, output_path: str,
                       sample_rate: int, channels: int) -> bool:
        """
        Converts an audio file to WAV format using ffmpeg.
        Returns True on success, False otherwise.
        """
        logger.info(f"Converting {input_path} to {output_path} (SR: {sample_rate}, Channels: {channels}).")
        try:
            subprocess.run(
                [
                    "ffmpeg", "-i", input_path,
                    "-ar", str(sample_rate),
                    "-ac", str(channels),
                    "-y", # Overwrite output file if it exists
                    output_path
                ],
                check=True, capture_output=True, text=True
            )
            logger.info(f"Successfully converted {input_path} to {output_path}.")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg conversion failed for {input_path}. Error: {e.stderr}")
            return False
        except FileNotFoundError:
            logger.error("ffmpeg command not found. Please ensure ffmpeg is installed and in PATH.")
            return False

# --- Whisper Transcription Service ---
class WhisperService:
    def __init__(self, model_name: str, device: torch.device):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        logger.info(f"Loading Whisper model '{self.model_name}' on device '{self.device}'.")
        try:
            self.model = whisper.load_model(self.model_name, device=self.device)
            logger.info(f"Whisper model '{self.model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model '{self.model_name}': {e}", exc_info=True)
            # Allow service to be created, but it won't be functional
            self.model = None 
            # Or raise an exception to prevent service creation if model is critical
            # raise RuntimeError(f"Failed to load Whisper model: {e}") from e


    def get_transcription_segments(self, audio_path: str) -> List[Dict[str, Any]]:
        if not self.model:
            logger.error("Whisper model not loaded. Cannot transcribe.")
            return []
        
        logger.info(f"Transcribing with Whisper: {audio_path} on device {self.device}")
        try:
            # fp16 is only for CUDA device
            use_fp16 = self.device.type == "cuda"
            result = self.model.transcribe(audio_path, fp16=use_fp16, word_timestamps=False)
            segments = result.get("segments", [])
            logger.info(f"Whisper transcription successful for: {audio_path}. Segments found: {len(segments)}")
            return segments
        except Exception as e:
            logger.error(f"Error during Whisper transcription for {audio_path}: {e}", exc_info=True)
            return []

# --- Pyannote Diarization Service ---
class PyannoteDiarizationService:
    def __init__(self, pipeline_name: str, auth_token: Optional[str], device: torch.device):
        self.pipeline_name = pipeline_name
        self.auth_token = auth_token
        self.device = device # Pyannote pipeline handles device internally based on torch global device or its own logic
        self.pipeline: Optional[PyannotePipeline] = None
        self._load_pipeline()

    def _load_pipeline(self):
        logger.info(f"Loading Pyannote diarization pipeline '{self.pipeline_name}'. Device hint: '{self.device}'")
        if not self.auth_token:
            logger.warning(f"Hugging Face auth token not provided for Pyannote pipeline '{self.pipeline_name}'. "
                           "This may fail if the model is gated.")
        try:
            self.pipeline = PyannotePipeline.from_pretrained(
                self.pipeline_name,
                use_auth_token=self.auth_token
            )
            # Check if the pipeline was actually loaded
            if self.pipeline is not None:
                 logger.info(f"Pyannote diarization pipeline '{self.pipeline_name}' appears to be loaded.")
            # Pyannote pipelines are usually composed of models that should be on the correct device.
            # If explicit movement is needed: self.pipeline.to(self.device)
            # For now, rely on pyannote's internal device handling.
            # No "successfully loaded" log here if from_pretrained raises an error,
            # as self.pipeline would not be set or would be None.
        except Exception as e:
            # This block will catch errors from from_pretrained, including auth/download issues
            logger.error(f"Failed to load Pyannote diarization pipeline '{self.pipeline_name}': {e}", exc_info=True)
            self.pipeline = None # Ensure it's None on failure
            # raise RuntimeError(f"Failed to load Pyannote pipeline: {e}") from e


    def get_speaker_turns(self, audio_path: str) -> List[Dict[str, Any]]:
        if not self.pipeline:
            logger.error("Pyannote diarization pipeline not loaded. Cannot diarize.")
            return []

        logger.info(f"Performing speaker diarization with Pyannote for: {audio_path}")
        try:
            diarization_result: PyannoteAnnotation = self.pipeline(audio_path)
            speaker_turns = []
            for turn, _, speaker_label in diarization_result.itertracks(yield_label=True):
                speaker_turns.append({
                    "speaker": speaker_label,
                    "start": turn.start,
                    "end": turn.end
                })
            logger.info(f"Pyannote diarization successful for: {audio_path}. Turns found: {len(speaker_turns)}")
            return speaker_turns
        except Exception as e:
            logger.error(f"Error during Pyannote diarization for {audio_path}: {e}", exc_info=True)
            return []

# --- Main Orchestration Service ---
class AudioProcessingService:
    def __init__(self, config: TranscriptionConfig):
        self.config = config
        self.config.device_str = DeviceManager.get_device_str() # Set the determined device string in config
        self.torch_device = DeviceManager.get_torch_device()

        logger.info(f"Initializing AudioProcessingService with device: {self.config.device_str}")

        self.whisper_service = WhisperService(
            model_name=self.config.whisper_model_name,
            device=self.torch_device
        )
        self.diarization_service = PyannoteDiarizationService(
            pipeline_name=self.config.diarization_pipeline_name,
            auth_token=self.config.hf_auth_token,
            device=self.torch_device # Pass device hint
        )
        self.audio_converter = AudioConverter() # Static methods, instantiation is optional

        if not self.whisper_service.model:
            logger.critical("Whisper service failed to initialize its model. Transcription will not work.")
        if not self.diarization_service.pipeline:
            logger.warning("Pyannote diarization service failed to initialize its pipeline. Diarization may not work or will be skipped.")


    def _merge_results(self, whisper_segments: List[Dict[str, Any]],
                       diarization_turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not diarization_turns:
            logger.warning("No diarization turns available. Returning segments with 'UNKNOWN' speaker.")
            return [{**seg, "speaker": "UNKNOWN"} for seg in whisper_segments]

        merged_output = []
        for seg in whisper_segments:
            seg_start = seg.get("start", 0.0) # Default if not present
            seg_end = seg.get("end", 0.0)   # Default if not present
            
            best_speaker = "UNKNOWN"
            max_overlap = 0.0

            for turn in diarization_turns:
                turn_start = turn["start"]
                turn_end = turn["end"]
                
                overlap_start = max(seg_start, turn_start)
                overlap_end = min(seg_end, turn_end)
                overlap_duration = max(0, overlap_end - overlap_start)

                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    best_speaker = turn["speaker"]
            
            merged_output.append({
                "speaker": best_speaker,
                "start": seg_start,
                "end": seg_end,
                "text": seg.get("text", "").strip()
            })
        return merged_output

    def transcribe_and_diarize(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Orchestrates the full transcription and diarization process.
        Manages temporary file creation and cleanup for WAV conversion.
        The original audio_path (e.g. uploaded .m4a) is also cleaned up.
        """
        if not self.whisper_service.model:
            logger.error("Whisper model not available in AudioProcessingService. Cannot process.")
            # Clean up original file if it exists, as processing cannot proceed
            if os.path.exists(audio_path):
                try: os.remove(audio_path); logger.info(f"Cleaned up {audio_path} as Whisper model is unavailable.")
                except Exception as e_rem: logger.error(f"Error cleaning up {audio_path}: {e_rem}")
            return []

        temp_wav_path: Optional[str] = None
        original_audio_path_for_cleanup = audio_path # Keep a reference for the finally block

        try:
            logger.info(f"AudioProcessingService: Starting process for {audio_path}")
            
            whisper_segments = self.whisper_service.get_transcription_segments(audio_path)
            
            if not whisper_segments:
                logger.warning(f"Whisper returned no segments for {audio_path}. Processing stopped.")
                return []

            diarization_turns: List[Dict[str, Any]] = []
            if self.diarization_service.pipeline:
                # Prepare path for converted WAV file
                base, _ = os.path.splitext(audio_path)
                temp_wav_path = base + "_converted_for_diarization.wav"

                conversion_success = self.audio_converter.convert_to_wav(
                    input_path=audio_path,
                    output_path=temp_wav_path,
                    sample_rate=self.config.audio_convert_sample_rate,
                    channels=self.config.audio_convert_channels
                )

                if conversion_success:
                    diarization_turns = self.diarization_service.get_speaker_turns(temp_wav_path)
                else:
                    logger.warning(f"Audio conversion to WAV failed for {audio_path}. Proceeding without diarization.")
            else:
                logger.warning("Diarization pipeline not available. Proceeding without diarization.")
            
            final_result = self._merge_results(whisper_segments, diarization_turns)
            logger.info(f"AudioProcessingService: Successfully processed {audio_path}")
            return final_result
            
        except Exception as e:
            logger.error(f"Error in AudioProcessingService for {audio_path}: {e}", exc_info=True)
            return [] # Return empty list on error, or re-raise if preferred
        finally:
            # Clean up the original uploaded temporary audio file
            if os.path.exists(original_audio_path_for_cleanup):
                try:
                    os.remove(original_audio_path_for_cleanup)
                    logger.info(f"Original temporary file {original_audio_path_for_cleanup} removed by AudioProcessingService.")
                except Exception as e_remove:
                    logger.error(f"Error removing original temporary file {original_audio_path_for_cleanup}: {e_remove}")
            
            # Clean up the temporary WAV file if it was created
            if temp_wav_path and os.path.exists(temp_wav_path):
                try:
                    os.remove(temp_wav_path)
                    logger.info(f"Temporary WAV file {temp_wav_path} removed by AudioProcessingService.")
                except Exception as e_remove:
                    logger.error(f"Error removing temporary WAV file {temp_wav_path}: {e_remove}")

# Example of how this might be initialized and used by app/main.py (conceptual)
# if __name__ == '__main__':
#     # This block is for illustration and direct testing if needed.
#     # In a real app, app/main.py would manage the lifecycle.
#     
#     # 1. Create Config (ensure HF_AUTH_TOKEN is set as env var or directly)
#     config = TranscriptionConfig(hf_auth_token=os.getenv("HF_AUTH_TOKEN_PYTEST", "YOUR_HF_TOKEN_HERE"))
# 
#     # 2. Initialize the main service
#     # This will also print device info and model loading logs
#     try:
#         processing_service = AudioProcessingService(config)
# 
#         # 3. Create a dummy audio file for testing (replace with a real m4a or other file)
#         # Ensure ffmpeg is installed if you use a non-wav format.
#         # For this example, let's assume a dummy_audio.m4a exists.
#         # You'd need to create one, e.g., by recording a short m4a.
#         test_audio_file_original = "dummy_audio.m4a" 
#         test_audio_file_for_processing = "/tmp/dummy_audio_test.m4a" # Use /tmp for safety
# 
#         if os.path.exists(test_audio_file_original):
#             import shutil
#             shutil.copy(test_audio_file_original, test_audio_file_for_processing)
#             logger.info(f"--- Testing full pipeline with {test_audio_file_for_processing} ---")
#             
#             if processing_service.whisper_service.model and processing_service.diarization_service.pipeline:
#                 final_output = processing_service.transcribe_and_diarize(test_audio_file_for_processing)
#                 logger.info("--- Final Merged Output: ---")
#                 for item in final_output:
#                     logger.info(item)
#             else:
#                 logger.error("One or more models in AudioProcessingService failed to load. Cannot run test.")
#             # The test_audio_file_for_processing should be deleted by transcribe_and_diarize
#         else:
#             logger.warning(f"Test audio file {test_audio_file_original} not found. Skipping full pipeline test.")
# 
#     except RuntimeError as e:
#         logger.error(f"Failed to initialize AudioProcessingService or run test: {e}")
#     except Exception as e:
#         logger.error(f"An unexpected error occurred during test: {e}", exc_info=True)
