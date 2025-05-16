# Whisper Transcription FastAPI Service

This project provides a FastAPI service to transcribe audio files using OpenAI's Whisper model. It is containerized using Docker and supports NVIDIA GPU acceleration with a fallback to CPU.

## Prerequisites

- Docker installed (https://docs.docker.com/get-docker/)
- NVIDIA GPU drivers installed on the host machine (if using GPU acceleration)
- NVIDIA Container Toolkit installed on the host machine (if using GPU acceleration) (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Project Structure

```
transcriber_project/
├── app/
│   ├── __init__.py
│   ├── main.py         # FastAPI application, API endpoints
│   └── transcriber.py  # Whisper model loading and transcription logic
├── Dockerfile          # Instructions to build the Docker image
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Setup and Running

### 1. Build the Docker Image

Navigate to the `transcriber_project` directory in your terminal and run:

```bash
docker build -t whisper-transcriber .
```

The first build might take some time as it needs to download the base CUDA image, Python dependencies, and the Whisper model itself (which happens when the container first starts or during health check if not cached).

### 2. Run the Docker Container

**With NVIDIA GPU acceleration:**

```bash
docker run --gpus all -p 8000:8000 --name whisper-app whisper-transcriber
```

**CPU only (if no NVIDIA GPU or NVIDIA Container Toolkit is not set up):**
The application will automatically fall back to CPU if CUDA is not detected inside the container.

```bash
docker run -p 8000:8000 --name whisper-app whisper-transcriber
```
*(Note: Performance will be significantly slower on CPU.)*

The container will start, and the FastAPI application will be available on port 8000. The first time the container runs, the Whisper model (`small.en`) will be downloaded if it hasn't been cached during the build (or a previous run). This can take a few minutes. Subsequent starts will be faster.

### 3. Check Application Health

Once the container is running, you can check its health status:

```bash
curl http://localhost:8000/health
```

Expected output (on successful GPU load):
```json
{
  "status": "ok",
  "message": "Application is healthy.",
  "model_status": "loaded",
  "device_used": "cuda",
  "whisper_model": "small.en"
}
```
Or (on successful CPU load):
```json
{
  "status": "ok",
  "message": "Application is healthy.",
  "model_status": "loaded",
  "device_used": "cpu",
  "whisper_model": "small.en"
}
```
If there was an error loading the model, the status will indicate an error.

### 4. Transcribe Audio

Send a POST request with an audio file to the `/transcribe/` endpoint.

Using `curl`:
```bash
curl -X POST -F "file=@/path/to/your/audiofile.mp3" http://localhost:8000/transcribe/
```
Replace `/path/to/your/audiofile.mp3` with the actual path to your audio file (e.g., `.wav`, `.m4a`, `.ogg` are also generally supported).

The response will be the plain text transcription.

Example:
If `audiofile.mp3` contains "Hello world", the output will be:
```
Hello world.
```

### 5. View Logs

To see the application logs from the container:
```bash
docker logs whisper-app
```
To follow the logs in real-time:
```bash
docker logs -f whisper-app
```

### 6. Stop and Remove the Container

To stop the container:
```bash
docker stop whisper-app
```
To remove the container (after stopping):
```bash
docker rm whisper-app
```

## Notes

- **Model Caching:** The Whisper model is downloaded by the `openai-whisper` library. It's typically cached in `~/.cache/whisper` within the user's home directory inside the container (e.g., `/home/appuser/.cache/whisper` for the `appuser`). If the container is removed and rebuilt without Docker layer caching for this step, the model might be re-downloaded.
- **Temporary Files:** Uploaded audio files are temporarily stored in the `/tmp/transcriber_uploads` directory inside the container and are deleted after transcription.
- **Error Handling:** The API includes basic error handling. If transcription fails or an invalid file is sent, it should return an appropriate HTTP error.
