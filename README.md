# Profanity Detection in Speech and Text

A robust multimodal system for detecting and rephrasing profanity in both speech and text, leveraging advanced NLP models to ensure accurate filtering while preserving conversational context.

![Profanity Detection System](https://img.shields.io/badge/AI-NLP%20System-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)

## 📋 Features

- **Multimodal Analysis**: Process both written text and spoken audio
- **Context-Aware Detection**: Goes beyond simple keyword matching
- **Automatic Content Refinement**: Intelligently rephrases content while preserving meaning
- **Audio Synthesis**: Converts rephrased content into high-quality spoken audio
- **Classification System**: Categorises content by toxicity levels
- **User-Friendly Interface**: Intuitive Gradio-based UI
- **Real-time Streaming**: Process audio in real-time as you speak
- **Adjustable Sensitivity**: Fine-tune profanity detection threshold
- **Visual Highlighting**: Instantly identify problematic words with visual highlighting
- **Toxicity Classification**: Automatically categorize content from "No Toxicity" to "Severe Toxicity"
- **Performance Optimization**: Half-precision support for improved GPU memory efficiency

## 🧠 Models Used

The system leverages four powerful models:

1. **Profanity Detection**: `parsawar/profanity_model_3.1` - A RoBERTa-based model trained for offensive language detection
2. **Content Refinement**: `s-nlp/t5-paranmt-detox` - A T5-based model for rephrasing offensive language
3. **Speech-to-Text**: OpenAI's `Whisper` (large) - For transcribing spoken audio
4. **Text-to-Speech**: Microsoft's `SpeechT5` - For converting rephrased text back to audio

## 🔧 Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU recommended (but CPU mode works too)
- FFmpeg for audio processing

### Option 1: Using Conda (Recommended for Local Development)

```bash
# Clone the repository
git clone https://github.com/yourusername/profanity-detection.git
cd profanity-detection

# Method A: Create environment from environment.yml (recommended)
conda env create -f environment.yml
conda activate llm_project

# Method B: Create a new conda environment manually
conda create -n profanity-detection python=3.10
conda activate profanity-detection

# Install PyTorch with CUDA support (adjust CUDA version if needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install FFmpeg for audio processing
conda install -c conda-forge ffmpeg

# Install Pillow properly to avoid DLL errors
conda install -c conda-forge pillow

# Install additional dependencies
pip install -r requirements.txt

# Set environment variable to avoid OpenMP conflicts (recommended)
conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE
conda activate profanity-detection  # Re-activate to apply the variable
```

### Option 2: Using Docker

```bash
# Clone the repository
git clone https://github.com/yourusername/profanity-detection.git
cd profanity-detection

# Build and run the Docker container
docker-compose build --no-cache

docker-compose up
```

## 🚀 Usage

### Running the Application

```bash
# Set environment variable to avoid OpenMP conflicts (if not set in conda config)
# For Windows:
set KMP_DUPLICATE_LIB_OK=TRUE

# For Linux/Mac:
export KMP_DUPLICATE_LIB_OK=TRUE

# Run the application
python profanity_detector.py
```

The Gradio interface will be accessible at http://127.0.0.1:7860 in your browser.

### Using the Interface

1. **Initialise Models**
   - Click the "Initialize Models" button when you first open the interface
   - Wait for all models to load (this may take a few minutes on first run)

2. **Text Analysis Tab**
   - Enter text into the text box
   - Adjust the "Profanity Detection Sensitivity" slider if needed
   - Click "Analyze Text"
   - View results including profanity score, toxicity classification, and rephrased content
   - See highlighted profane words in the text
   - Listen to the audio version of the rephrased content

3. **Audio Analysis Tab**
   - Upload an audio file or record directly using your microphone
   - Click "Analyze Audio"
   - View transcription, profanity analysis, and rephrased content
   - Listen to the cleaned audio version of the rephrased content

4. **Real-time Streaming Tab**
   - Click "Start Real-time Processing"
   - Speak into your microphone
   - Watch as your speech is transcribed, analyzed, and rephrased in real-time
   - Listen to the clean audio output
   - Click "Stop Real-time Processing" when finished

## 🔧 Deployment Options

### Local Deployment with Conda

For the best development experience with fine-grained control:

```bash
# Create and configure environment
conda env create -f environment.yml
conda activate llm_project

# Run with sharing enabled (accessible from other devices)
python profanity_detector.py
```

### Docker Deployment (Production)

For containerised deployment with predictable environment:

#### Basic CPU Deployment
```bash
docker-compose up --build
```

#### GPU-Accelerated Deployment
```bash
# Automatic detection (recommended)
docker-compose up --build

# Or explicitly request GPU mode
docker-compose up --build profanity-detector-gpu
```

No need to edit any configuration files - the system will automatically detect and use your GPU if available.

#### Custom Port Configuration
To change the default port (7860):
1. Edit docker-compose.yml and change the port mapping (e.g., "8080:7860")
2. Run `docker-compose up --build`

## ⚠️ Troubleshooting

### OpenMP Runtime Conflict

If you encounter this error:
```
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```

**Solutions:**

1. **Temporary fix**: Set environment variable before running:
   ```bash
   set KMP_DUPLICATE_LIB_OK=TRUE  # Windows
   export KMP_DUPLICATE_LIB_OK=TRUE  # Linux/Mac
   ```

2. **Code-based fix**: Add to the beginning of your script:
   ```python
   import os
   os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
   ```

3. **Permanent fix for Conda environment**:
   ```bash
   conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE -n profanity-detection
   conda deactivate
   conda activate profanity-detection
   ```

### GPU Memory Issues

If you encounter CUDA out of memory errors:

1. Use smaller models:
   ```python
   # Change Whisper from "large" to "medium" or "small"
   whisper_model = whisper.load_model("medium").to(device)
   
   # Keep the TTS model on CPU to save GPU memory
   tts_model = SpeechT5ForTextToSpeech.from_pretrained(TTS_MODEL)  # CPU mode
   ```

2. Run some models on CPU instead of GPU:
   ```python
   # Remove .to(device) to keep model on CPU
   t5_model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL)  # CPU mode
   ```

3. Use Docker with specific GPU memory limits:
   ```yaml
   # In docker-compose.yml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: 1
             capabilities: [gpu]
             options:
               memory: 4G  # Limit to 4GB of GPU memory
   ```

### Docker-Specific Issues

1. **Permission issues with mounted volumes**:
   ```bash
   # Fix permissions (Linux/Mac)
   sudo chown -R $USER:$USER .
   ```

2. **No GPU access in container**:
   - Verify NVIDIA Container Toolkit installation
   - Check GPU driver compatibility
   - Run `nvidia-smi` on the host to confirm GPU availability

### First-Time Slowness

When first run, the application downloads all models, which may take time. Subsequent runs will be faster as models are cached locally. The text-to-speech model requires additional download time on first use.

## 📄 Project Structure

```
profanity-detection/
├── profanity_detector.py    # Main application file
├── Dockerfile               # For containerised deployment
├── docker-compose.yml       # Container orchestration
├── requirements.txt         # Python dependencies
├── environment.yml          # Conda environment specification
└── README.md                # This file
```

## 📚 References

- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Microsoft SpeechT5](https://huggingface.co/microsoft/speecht5_tts)
- [Gradio Documentation](https://gradio.app/docs/)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- This project utilises models from HuggingFace Hub, Microsoft, and OpenAI
- Inspired by research in content moderation and responsible AI
