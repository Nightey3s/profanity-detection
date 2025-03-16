# Profanity Detection in Speech and Text

A robust multimodal system for detecting and rephrasing profanity in both speech and text, leveraging advanced NLP models to ensure accurate filtering while preserving conversational context.

![Profanity Detection System](https://img.shields.io/badge/AI-NLP%20System-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)

## 🌐 Live Demo

Try the system without installation via our Hugging Face Spaces deployment:

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/nightey3s/profanity-detection)

<img src="https://briantham.com/assets/img/projects/qr-code/Profanity-Detection-huggingface-qr-code.svg?sanitize=true" alt="QR Code" width="300" />

This live version leverages Hugging Face's ZeroGPU technology, which provides on-demand GPU acceleration for inference while optimising resource usage.

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
- **Cloud Deployment**: Available as a hosted service on Hugging Face Spaces

## 🧠 Models Used

The system leverages four powerful models:

1. **Profanity Detection**: `parsawar/profanity_model_3.1` - A RoBERTa-based model trained for offensive language detection
2. **Content Refinement**: `s-nlp/t5-paranmt-detox` - A T5-based model for rephrasing offensive language
3. **Speech-to-Text**: OpenAI's `Whisper` (large-v2) - For transcribing spoken audio
4. **Text-to-Speech**: Microsoft's `SpeechT5` - For converting rephrased text back to audio

## 🚀 Deployment Options

### Online Deployment (No Installation Required)

Access the application directly through Hugging Face Spaces:
- **URL**: [https://huggingface.co/spaces/nightey3s/profanity-detection](https://huggingface.co/spaces/nightey3s/profanity-detection)
- **Technology**: Built with ZeroGPU for efficient GPU resource allocation
- **Features**: All features of the full application accessible through your browser
- **Source Code**: [GitHub Repository](https://github.com/Nightey3s/profanity-detection)

### Local Installation

#### Prerequisites

- Python 3.10+
- CUDA-compatible GPU recommended (but CPU mode works too)
- FFmpeg for audio processing

#### Option 1: Using Conda (Recommended for Local Development)

```bash
# Clone the repository
git clone https://github.com/Nightey3s/profanity-detection.git
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

#### Option 2: Using Docker

```bash
# Clone the repository
git clone https://github.com/Nightey3s/profanity-detection.git
cd profanity-detection

# Build and run the Docker container
docker-compose build --no-cache

docker-compose up
```

## 🔧 Usage

### Using the Online Interface (Hugging Face Spaces)

1. Visit [https://huggingface.co/spaces/nightey3s/profanity-detection](https://huggingface.co/spaces/nightey3s/profanity-detection)
2. The interface might take a moment to load on first access as it allocates resources
3. Follow the same usage instructions as below, starting with "Initialize Models"

### Using the Local Interface

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

### Hugging Face Spaces-Specific Issues

1. **Long initialization time**: The first time you access the Space, it may take longer to initialize as models are downloaded and cached.

2. **Timeout errors**: If the model takes too long to process your request, try again with shorter text or audio inputs.

3. **Browser compatibility**: Ensure your browser allows microphone access for audio recording features.

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

## Team Members

- Brian Tham
- Hong Ziyang
- Nabil Zafran
- Adrian Ian Wong
- Lin Xiang Hong

## 📚 References

- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Microsoft SpeechT5](https://huggingface.co/microsoft/speecht5_tts)
- [Gradio Documentation](https://gradio.app/docs/)
- [Hugging Face Spaces](https://huggingface.co/spaces)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- This project utilises models from HuggingFace Hub, Microsoft, and OpenAI
- Inspired by research in content moderation and responsible AI
- Hugging Face for providing the Spaces platform with ZeroGPU technology