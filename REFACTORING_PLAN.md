# Plan: Restructuring `app/transcriber.py`

## 1. Goal

To refactor the `app/transcriber.py` module to improve its structure, maintainability, extensibility, and debuggability. This will be achieved by moving from a script with global variables and procedural functions to a more object-oriented approach using classes, each with a specific responsibility.

## 2. Proposed Class Structure

### 2.1. `TranscriptionConfig` Class
A dataclass or simple class to hold all configuration parameters.
*   **Attributes:**
    *   `WHISPER_MODEL_NAME: str` (e.g., "small.en")
    *   `DIARIZATION_PIPELINE_NAME: str` (e.g., "pyannote/speaker-diarization-3.1")
    *   `HF_AUTH_TOKEN: str` (Hugging Face auth token - **Note:** This should ideally be moved to an environment variable in a future iteration for better security).
    *   `AUDIO_CONVERT_SAMPLE_RATE: int` (e.g., 16000)
    *   `AUDIO_CONVERT_CHANNELS: int` (e.g., 1)
*   **Purpose:** Centralizes configuration, making it easier to manage and modify.

### 2.2. `DeviceManager` Utility
A helper function or a small utility class.
*   **Method:** `get_device() -> torch.device`
*   **Purpose:** Determines and returns the appropriate `torch.device` (e.g., `torch.device("cuda")` or `torch.device("cpu")`) based on system capabilities (e.g., `torch.cuda.is_available()`).

### 2.3. `WhisperService` Class
Handles all Whisper-related operations.
*   **`__init__(self, model_name: str, device: torch.device)`:**
    *   Loads the specified Whisper model onto the given device.
    *   Stores the loaded model as an instance variable.
*   **`get_transcription_segments(self, audio_path: str) -> list`:**
    *   Takes an audio path.
    *   Uses the loaded Whisper model to transcribe and return detailed segments (including timestamps).

### 2.4. `PyannoteDiarizationService` Class
Handles all `pyannote.audio` diarization operations.
*   **`__init__(self, pipeline_name: str, auth_token: str, device: torch.device)`:**
    *   Loads the specified `pyannote.audio` pipeline using the auth token.
    *   Ensures the pipeline components are on the correct device.
    *   Stores the loaded pipeline as an instance variable.
*   **`get_speaker_turns(self, audio_path: str) -> list`:**
    *   Takes an audio path (expected to be in a compatible format like WAV).
    *   Uses the loaded pipeline to perform diarization and return speaker turns (e.g., `[{'speaker': 'SPEAKER_00', 'start': 0.5, 'end': 2.3}, ...]`).

### 2.5. `AudioConverter` Utility
Handles audio format conversions.
*   **Method (static or instance):** `convert_to_wav(self, input_path: str, output_path: str, sample_rate: int, channels: int) -> bool`
    *   Encapsulates the `ffmpeg` command-line logic to convert an audio file to WAV format with specified parameters.
    *   Returns `True` on successful conversion, `False` otherwise, and logs errors.

### 2.6. `AudioProcessingService` Class (Orchestrator)
The main service class that coordinates the transcription and diarization process.
*   **`__init__(self, config: TranscriptionConfig)`:**
    *   Initializes the `DeviceManager` (or calls its utility function) to get the `torch.device`.
    *   Instantiates `WhisperService` with the model name from `config` and the determined device.
    *   Instantiates `PyannoteDiarizationService` with the pipeline name and auth token from `config`, and the determined device.
    *   Instantiates `AudioConverter` (or makes its methods available).
*   **`transcribe_and_diarize(self, audio_path: str) -> list`:**
    *   This will be the primary public method called by `app/main.py`.
    *   **Workflow:**
        1.  Call `WhisperService.get_transcription_segments()` on the original `audio_path`.
        2.  If transcription is successful and segments are produced:
            *   Generate a path for a temporary WAV file (e.g., in the same directory as `audio_path` or a dedicated temp location).
            *   Call `AudioConverter.convert_to_wav()` to convert the original `audio_path` to the temporary WAV path, using sample rate and channels from `config`.
            *   If WAV conversion is successful, call `PyannoteDiarizationService.get_speaker_turns()` on the temporary WAV path.
            *   If WAV conversion fails, log a warning and proceed with empty diarization turns (or handle error as per policy).
        3.  Else (if Whisper returns no segments), log and return an appropriate empty result.
        4.  Call a private helper method `_merge_results(whisper_segments, diarization_turns)` to combine the outputs.
        5.  Return the merged results (e.g., `[{'speaker': 'A', 'start': 0.5, 'end': 2.3, 'text': 'Hello.'}, ...]`).
    *   **File Management:** This method is responsible for managing the lifecycle of the temporary WAV file (creation and deletion in a `finally` block). It will also be responsible for deleting the original `audio_path` file after all processing is complete.
*   **`_merge_results(self, whisper_segments: list, diarization_turns: list) -> list`:** (Can be a private helper or static method)
    *   Contains the logic to combine Whisper segments with speaker turns based on timestamp overlap (similar to the current implementation).

## 3. Impact on `app/main.py`

*   The `lifespan` context manager in `app/main.py` will:
    *   Create an instance of `TranscriptionConfig` (potentially loading some values from environment variables).
    *   Instantiate the `AudioProcessingService` once with this config.
    *   Store the `AudioProcessingService` instance in `app_state`.
*   The `/transcribe` endpoint in `app/main.py` will call the `transcribe_and_diarize` method on the stored `AudioProcessingService` instance.
*   The `/health` endpoint in `app/main.py` will be simplified, as the readiness of the `AudioProcessingService` (which implies its sub-services are ready) will be the main indicator. Detailed model loading errors can still be propagated from the service.
*   Global variables for models and pipelines (`_model`, `_device`, `_diarization_pipeline`) in the current `app/transcriber.py` will be removed, as their state will be managed within the respective class instances.

## 4. Benefits

*   **Separation of Concerns:** Each class has a distinct, well-defined responsibility.
*   **Improved Testability:** Individual services (`WhisperService`, `PyannoteDiarizationService`, `AudioConverter`) can be unit-tested more easily, potentially with mocks for their dependencies.
*   **Better State Management:** Models, pipelines, and configurations are encapsulated within class instances, avoiding reliance on global variables.
*   **Enhanced Readability and Maintainability:** The code structure becomes more logical and easier to follow.
*   **Easier Extensibility:** Adding new functionalities (e.g., a different diarization tool, another ASR engine) can be done by creating new service classes that conform to an expected interface, and the `AudioProcessingService` can be adapted or configured to use them.

## 5. Diagram

```mermaid
graph TD
    A[app/main.py] -- Manages Lifecycle & Routes Calls --> B(AudioProcessingService Instance);

    subgraph app.transcriber_restructured
        C[TranscriptionConfig] -- Provides Config --> B;
        D[DeviceManager Utility] -- Determines Device --> B;
        B -- Uses --> E(WhisperService);
        B -- Uses --> F(PyannoteDiarizationService);
        B -- Uses --> G(AudioConverter Utility);
        
        E -- Loads & Uses --> H[Whisper Model];
        F -- Loads & Uses --> I[Pyannote Pipeline];
        G -- Uses --> J[ffmpeg CLI];
    end

    K[Uploaded Audio File] -- Input --> B;
    B -- Returns --> L[Diarized Transcription];

    B -- Manages --> M[Temp WAV File (created & deleted)];
    B -- Manages --> N[Original Uploaded File (deleted after processing)];
```

## 6. Next Steps

1.  Obtain user approval for this refactoring plan.
2.  If approved, switch to "Code" mode.
3.  Implement the new class structure in `app/transcriber.py`.
4.  Update `app/main.py` to use the new `AudioProcessingService`.
5.  Thoroughly test the refactored application.