import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import whisper
import gradio as gr
import re
import pandas as pd
import numpy as np
import os
import time
import logging
import threading
import queue
from scipy.io.wavfile import write as write_wav
from html import escape
import traceback
import spaces # Required for Hugging Face ZeroGPU compatibility

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('profanity_detector')

# Detect if we're running in a ZeroGPU environment
IS_ZEROGPU = os.environ.get("SPACE_RUNTIME_STATELESS", "0") == "1"
if os.environ.get("SPACES_ZERO_GPU") is not None:
    IS_ZEROGPU = True

# Define device strategy that works in both environments
if IS_ZEROGPU:
    # In ZeroGPU: always initialize on CPU, will use GPU only in @spaces.GPU functions
    device = torch.device("cpu")
    logger.info("ZeroGPU environment detected. Using CPU for initial loading.")
else:
    # For local runs: use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Local environment. Using device: {device}")

# Global variables for models
profanity_model = None
profanity_tokenizer = None
t5_model = None
t5_tokenizer = None
whisper_model = None
tts_processor = None
tts_model = None
vocoder = None
models_loaded = False

# Default speaker embeddings for TTS
speaker_embeddings = None

# Queue for real-time audio processing
audio_queue = queue.Queue()
processing_active = False

# Model loading with int8 quantization
def load_models():
    global profanity_model, profanity_tokenizer, t5_model, t5_tokenizer, whisper_model
    global tts_processor, tts_model, vocoder, speaker_embeddings, models_loaded
    
    try:
        logger.info("Loading profanity detection model...")
        PROFANITY_MODEL = "parsawar/profanity_model_3.1"
        profanity_tokenizer = AutoTokenizer.from_pretrained(PROFANITY_MODEL)
        
        # Load model without moving to CUDA directly
        profanity_model = AutoModelForSequenceClassification.from_pretrained(
            PROFANITY_MODEL,
            device_map=None,  # Stay on CPU for now
            low_cpu_mem_usage=True
        )
        
        # Only move to device if NOT in ZeroGPU mode
        if not IS_ZEROGPU and torch.cuda.is_available():
            profanity_model = profanity_model.to(device)
            try:
                profanity_model = profanity_model.half()
                logger.info("Successfully converted profanity model to half precision")
            except Exception as e:
                logger.warning(f"Could not convert to half precision: {str(e)}")
        
        logger.info("Loading detoxification model...")
        T5_MODEL = "s-nlp/t5-paranmt-detox"
        t5_tokenizer = AutoTokenizer.from_pretrained(T5_MODEL)
        
        t5_model = AutoModelForSeq2SeqLM.from_pretrained(
            T5_MODEL,
            device_map=None,  # Stay on CPU for now
            low_cpu_mem_usage=True
        )
        
        # Only move to device if NOT in ZeroGPU mode
        if not IS_ZEROGPU and torch.cuda.is_available():
            t5_model = t5_model.to(device)
            try:
                t5_model = t5_model.half()
                logger.info("Successfully converted T5 model to half precision")
            except Exception as e:
                logger.warning(f"Could not convert to half precision: {str(e)}")
        
        logger.info("Loading Whisper speech-to-text model...")
        # Always load on CPU in ZeroGPU mode
        #whisper_model = whisper.load_model("medium" if IS_ZEROGPU else "large", device="cpu")
        whisper_model = whisper.load_model("large-v2", device="cpu")
        
        # Only move to device if NOT in ZeroGPU mode
        if not IS_ZEROGPU and torch.cuda.is_available():
            whisper_model = whisper_model.to(device)
            
        logger.info("Loading Text-to-Speech model...")
        TTS_MODEL = "microsoft/speecht5_tts"
        tts_processor = SpeechT5Processor.from_pretrained(TTS_MODEL)
        
        tts_model = SpeechT5ForTextToSpeech.from_pretrained(
            TTS_MODEL,
            device_map=None,  # Stay on CPU for now
            low_cpu_mem_usage=True
        )
        
        vocoder = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan",
            device_map=None,  # Stay on CPU for now
            low_cpu_mem_usage=True
        )
        
        # Only move to device if NOT in ZeroGPU mode
        if not IS_ZEROGPU and torch.cuda.is_available():
            tts_model = tts_model.to(device)
            vocoder = vocoder.to(device)
        
        # Speaker embeddings - always on CPU for ZeroGPU
        speaker_embeddings = torch.zeros((1, 512))
        # Only move to device if NOT in ZeroGPU mode
        if not IS_ZEROGPU and torch.cuda.is_available():
            speaker_embeddings = speaker_embeddings.to(device)
            
        models_loaded = True
        logger.info("All models loaded successfully.")
        
        return "Models loaded successfully."
    except Exception as e:
        error_msg = f"Error loading models: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return error_msg

# ZeroGPU decorator: Requests GPU resources when function is called and releases them when completed.
# This enables efficient GPU sharing in Hugging Face Spaces while having no effect in local environments.
@spaces.GPU
def detect_profanity(text: str, threshold: float = 0.5):
    """
    Detect profanity in text with adjustable threshold
    
    Args:
        text: The input text to analyze
        threshold: Profanity detection threshold (0.0-1.0)
        
    Returns:
        Dictionary with analysis results
    """
    if not models_loaded:
        return {"error": "Models not loaded yet. Please wait."}
    
    try:
        # Detect profanity and score
        inputs = profanity_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # In ZeroGPU, move to GPU here inside the spaces.GPU function
        # For local environments, it might already be on the correct device
        current_device = device
        if IS_ZEROGPU and torch.cuda.is_available():
            current_device = torch.device("cuda")
            inputs = inputs.to(current_device)
            # Only in ZeroGPU mode, we need to move the model to GPU inside the function
            profanity_model.to(current_device)
        elif torch.cuda.is_available():  # Local environment with CUDA
            inputs = inputs.to(current_device)
            
        with torch.no_grad():
            outputs = profanity_model(**inputs).logits
        score = torch.nn.functional.softmax(outputs, dim=1)[0][1].item()

        # Identify specific profane words
        words = re.findall(r'\b\w+\b', text)
        profane_words = []
        word_scores = {}
        
        if score > threshold:
            for word in words:
                if len(word) < 2:  # Skip very short words
                    continue
                    
                word_inputs = profanity_tokenizer(word, return_tensors="pt", truncation=True, max_length=512)
                if torch.cuda.is_available():
                    word_inputs = word_inputs.to(current_device)
                    
                with torch.no_grad():
                    word_outputs = profanity_model(**word_inputs).logits
                word_score = torch.nn.functional.softmax(word_outputs, dim=1)[0][1].item()
                word_scores[word] = word_score
                
                if word_score > threshold:
                    profane_words.append(word.lower())

        # Move model back to CPU if in ZeroGPU mode - to free GPU memory
        if IS_ZEROGPU and torch.cuda.is_available():
            profanity_model.to(torch.device("cpu"))

        # Create highlighted version of the text
        highlighted_text = create_highlighted_text(text, profane_words)

        return {
            "text": text, 
            "score": score, 
            "profanity": score > threshold, 
            "profane_words": profane_words,
            "highlighted_text": highlighted_text,
            "word_scores": word_scores
        }
    except Exception as e:
        error_msg = f"Error in profanity detection: {str(e)}"
        logger.error(error_msg)
        # Make sure model is on CPU if in ZeroGPU mode - to free GPU memory
        if IS_ZEROGPU and torch.cuda.is_available():
            try:
                profanity_model.to(torch.device("cpu"))
            except:
                pass
        return {"error": error_msg, "text": text, "score": 0, "profanity": False}

def create_highlighted_text(text, profane_words):
    """
    Create HTML-formatted text with profane words highlighted
    """
    if not profane_words:
        return escape(text)
        
    # Create a regex pattern matching any of the profane words (case insensitive)
    pattern = r'\b(' + '|'.join(re.escape(word) for word in profane_words) + r')\b'
    
    # Replace occurrences with highlighted versions
    def highlight_match(match):
        return f'<span style="background-color: rgba(255, 0, 0, 0.3); padding: 0px 2px; border-radius: 3px;">{match.group(0)}</span>'
    
    highlighted = re.sub(pattern, highlight_match, text, flags=re.IGNORECASE)
    return highlighted

@spaces.GPU
def rephrase_profanity(text):
    """
    Rephrase text containing profanity
    """
    if not models_loaded:
        return "Models not loaded yet. Please wait."
        
    try:
        # Rephrase using the detoxification model
        inputs = t5_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # In ZeroGPU, move to GPU here inside the spaces.GPU function
        current_device = device
        if IS_ZEROGPU and torch.cuda.is_available():
            current_device = torch.device("cuda")
            inputs = inputs.to(current_device)
            # Only in ZeroGPU mode, we need to move the model to GPU inside the function
            t5_model.to(current_device)
        elif torch.cuda.is_available():  # Local environment with CUDA
            inputs = inputs.to(current_device)
        
        # Use more conservative generation settings with error handling
        try:
            outputs = t5_model.generate(
                **inputs,
                max_length=512,
                num_beams=4,         # Reduced from 5 to be more memory-efficient
                early_stopping=True,
                no_repeat_ngram_size=2,
                length_penalty=1.0
            )
            rephrased_text = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Verify the output is reasonable
            if not rephrased_text or len(rephrased_text) < 3:
                logger.warning(f"T5 model produced unusable output: '{rephrased_text}'")
                return text  # Return original if output is too short
                
            # Move model back to CPU if in ZeroGPU mode - to free GPU memory
            if IS_ZEROGPU and torch.cuda.is_available():
                t5_model.to(torch.device("cpu"))
                
            return rephrased_text.strip()
            
        except RuntimeError as e:
            # Handle potential CUDA out of memory error
            if "CUDA out of memory" in str(e):
                logger.warning("CUDA out of memory in T5 model. Trying with smaller beam size...")
                # Try again with smaller beam size
                outputs = t5_model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=2,  # Use smaller beam size
                    early_stopping=True
                )
                rephrased_text = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Move model back to CPU if in ZeroGPU mode - to free GPU memory
                if IS_ZEROGPU and torch.cuda.is_available():
                    t5_model.to(torch.device("cpu"))
                    
                return rephrased_text.strip()
            else:
                raise e  # Re-raise if it's not a memory issue
                
    except Exception as e:
        error_msg = f"Error in rephrasing: {str(e)}"
        logger.error(error_msg)
        # Make sure model is on CPU if in ZeroGPU mode - to free GPU memory
        if IS_ZEROGPU and torch.cuda.is_available():
            try:
                t5_model.to(torch.device("cpu"))
            except:
                pass
        return text  # Return original text if rephrasing fails

@spaces.GPU
def text_to_speech(text):
    """
    Convert text to speech using SpeechT5
    """
    if not models_loaded:
        return None
        
    try:
        # Create a temporary file path to save the audio
        temp_file = f"temp_tts_output_{int(time.time())}.wav"
        
        # Process the text input
        inputs = tts_processor(text=text, return_tensors="pt")
        
        # In ZeroGPU, move to GPU here inside the spaces.GPU function
        current_device = device
        if IS_ZEROGPU and torch.cuda.is_available():
            current_device = torch.device("cuda")
            inputs = inputs.to(current_device)
            # Only in ZeroGPU mode, we need to move the models to GPU inside the function
            tts_model.to(current_device)
            vocoder.to(current_device)
            speaker_embeddings_local = speaker_embeddings.to(current_device)
        elif torch.cuda.is_available():  # Local environment with CUDA
            inputs = inputs.to(current_device)
            speaker_embeddings_local = speaker_embeddings
        else:
            speaker_embeddings_local = speaker_embeddings
        
        # Generate speech with a fixed speaker embedding
        speech = tts_model.generate_speech(
            inputs["input_ids"], 
            speaker_embeddings_local, 
            vocoder=vocoder
        )
        
        # Convert from PyTorch tensor to NumPy array
        speech_np = speech.cpu().numpy()
        
        # Move models back to CPU if in ZeroGPU mode - to free GPU memory
        if IS_ZEROGPU and torch.cuda.is_available():
            tts_model.to(torch.device("cpu"))
            vocoder.to(torch.device("cpu"))
        
        # Save as WAV file (sampling rate is 16kHz for SpeechT5)
        write_wav(temp_file, 16000, speech_np)
        
        return temp_file
    except Exception as e:
        error_msg = f"Error in text-to-speech conversion: {str(e)}"
        logger.error(error_msg)
        # Make sure models are on CPU if in ZeroGPU mode - to free GPU memory
        if IS_ZEROGPU and torch.cuda.is_available():
            try:
                tts_model.to(torch.device("cpu"))
                vocoder.to(torch.device("cpu"))
            except:
                pass
        return None

def text_analysis(input_text, threshold=0.5):
    """
    Analyze text for profanity with adjustable threshold
    """
    if not models_loaded:
        return "Models not loaded yet. Please wait for initialization to complete.", None, None
        
    try:
        # Detect profanity with the given threshold
        result = detect_profanity(input_text, threshold=threshold)
        
        # Handle error case
        if "error" in result:
            return result["error"], None, None
            
        # Process results
        if result["profanity"]:
            clean_text = rephrase_profanity(input_text)
            profane_words_str = ", ".join(result["profane_words"])
            
            toxicity_score = result["score"]
            
            classification = (
                "Severe Toxicity" if toxicity_score >= 0.7 else
                "Moderate Toxicity" if toxicity_score >= 0.5 else
                "Mild Toxicity" if toxicity_score >= 0.35 else
                "Minimal Toxicity" if toxicity_score >= 0.2 else
                "No Toxicity"
            )
            
            # Generate audio for the rephrased text
            audio_output = text_to_speech(clean_text)
            
            return (
                f"Profanity Score: {result['score']:.4f}\n\n"
                f"Profane: {result['profanity']}\n"
                f"Classification: {classification}\n"
                f"Detected Profane Words: {profane_words_str}\n\n"
                f"Reworded: {clean_text}"
            ), result["highlighted_text"], audio_output
        else:
            # If no profanity detected, just convert the original text to speech
            audio_output = text_to_speech(input_text)
            
            return (
                f"Profanity Score: {result['score']:.4f}\n"
                f"Profane: {result['profanity']}\n"
                f"Classification: No Toxicity"
            ), None, audio_output
    except Exception as e:
        error_msg = f"Error in text analysis: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return error_msg, None, None

# ZeroGPU decorator with custom duration: Allocates GPU for up to 120 seconds to handle longer audio processing.
# Longer durations ensure processing isn't cut off, while shorter durations improve queue priority.
@spaces.GPU(duration=120)
def analyze_audio(audio_path, threshold=0.5):
    """
    Analyze audio for profanity with adjustable threshold
    """
    if not models_loaded:
        return "Models not loaded yet. Please wait for initialization to complete.", None, None
        
    if not audio_path:
        return "No audio provided.", None, None
        
    try:
        # In ZeroGPU mode, models need to be moved to GPU
        if IS_ZEROGPU and torch.cuda.is_available():
            current_device = torch.device("cuda")
            whisper_model.to(current_device)
        
        # Transcribe audio
        result = whisper_model.transcribe(audio_path, fp16=torch.cuda.is_available())
        text = result["text"]
        
        # Move whisper model back to CPU if in ZeroGPU mode
        if IS_ZEROGPU and torch.cuda.is_available():
            whisper_model.to(torch.device("cpu"))
        
        # Detect profanity with user-defined threshold
        analysis = detect_profanity(text, threshold=threshold)
        
        # Handle error case
        if "error" in analysis:
            return f"Error during analysis: {analysis['error']}\nTranscription: {text}", None, None

        if analysis["profanity"]:
            clean_text = rephrase_profanity(text)
        else:
            clean_text = text

        # Generate audio for the rephrased text
        audio_output = text_to_speech(clean_text)
        
        return (
            f"Transcription: {text}\n\n"
            f"Profanity Score: {analysis['score']:.4f}\n"
            f"Profane: {'Yes' if analysis['profanity'] else 'No'}\n"
            f"Classification: {'Severe Toxicity' if analysis['score'] >= 0.7 else 'Moderate Toxicity' if analysis['score'] >= 0.5 else 'Mild Toxicity' if analysis['score'] >= 0.35 else 'Minimal Toxicity' if analysis['score'] >= 0.2 else 'No Toxicity'}\n"
            f"Profane Words: {', '.join(analysis['profane_words']) if analysis['profanity'] else 'None'}\n\n"
            f"Reworded: {clean_text}"
        ), analysis["highlighted_text"] if analysis["profanity"] else None, audio_output
    except Exception as e:
        error_msg = f"Error in audio analysis: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        # Make sure models are on CPU if in ZeroGPU mode
        if IS_ZEROGPU and torch.cuda.is_available():
            try:
                whisper_model.to(torch.device("cpu"))
            except:
                pass
        return error_msg, None, None

# Global variables to store streaming results
stream_results = {
    "transcript": "",
    "profanity_info": "",
    "clean_text": "",
    "audio_output": None
}

@spaces.GPU
def process_stream_chunk(audio_chunk):
    """Process an audio chunk from the streaming interface"""
    global stream_results, processing_active
    
    if not processing_active or not models_loaded:
        return stream_results["transcript"], stream_results["profanity_info"], stream_results["clean_text"], stream_results["audio_output"]
    
    try:
        # The format of audio_chunk from Gradio streaming can vary
        # It can be: (numpy_array, sample_rate), (filepath, sample_rate, numpy_array) or just numpy_array
        # Let's handle all possible cases
        
        if audio_chunk is None:
            # No audio received
            return stream_results["transcript"], stream_results["profanity_info"], stream_results["clean_text"], stream_results["audio_output"]
        
        # Different Gradio versions return different formats
        temp_file = None
        
        if isinstance(audio_chunk, tuple):
            if len(audio_chunk) == 2:
                # Format: (numpy_array, sample_rate)
                samples, sample_rate = audio_chunk
                temp_file = f"temp_stream_{int(time.time())}.wav"
                write_wav(temp_file, sample_rate, samples)
            elif len(audio_chunk) == 3:
                # Format: (filepath, sample_rate, numpy_array)
                filepath, sample_rate, samples = audio_chunk
                # Use the provided filepath if it exists
                if os.path.exists(filepath):
                    temp_file = filepath
                else:
                    # Create our own file
                    temp_file = f"temp_stream_{int(time.time())}.wav"
                    write_wav(temp_file, sample_rate, samples)
        elif isinstance(audio_chunk, np.ndarray):
            # Just a numpy array, assume sample rate of 16000 for Whisper
            samples = audio_chunk
            sample_rate = 16000
            temp_file = f"temp_stream_{int(time.time())}.wav"
            write_wav(temp_file, sample_rate, samples)
        elif isinstance(audio_chunk, str) and os.path.exists(audio_chunk):
            # It's a filepath
            temp_file = audio_chunk
        else:
            # Unknown format
            stream_results["profanity_info"] = f"Error: Unknown audio format: {type(audio_chunk)}"
            return stream_results["transcript"], stream_results["profanity_info"], stream_results["clean_text"], stream_results["audio_output"]
        
        # Make sure we have a valid file to process
        if not temp_file or not os.path.exists(temp_file):
            stream_results["profanity_info"] = "Error: Failed to create audio file for processing"
            return stream_results["transcript"], stream_results["profanity_info"], stream_results["clean_text"], stream_results["audio_output"]
            
        # In ZeroGPU mode, move whisper model to GPU
        if IS_ZEROGPU and torch.cuda.is_available():
            current_device = torch.device("cuda")
            whisper_model.to(current_device)
            
        # Process with Whisper
        result = whisper_model.transcribe(temp_file, fp16=torch.cuda.is_available())
        transcript = result["text"].strip()
        
        # Move whisper model back to CPU if in ZeroGPU mode
        if IS_ZEROGPU and torch.cuda.is_available():
            whisper_model.to(torch.device("cpu"))
        
        # Skip processing if transcript is empty
        if not transcript:
            # Clean up temp file if we created it
            if temp_file and temp_file.startswith("temp_stream_") and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            # Return current state, but update profanity info
            stream_results["profanity_info"] = "No speech detected. Keep talking..."
            return stream_results["transcript"], stream_results["profanity_info"], stream_results["clean_text"], stream_results["audio_output"]
        
        # Update transcript
        stream_results["transcript"] = transcript
        
        # Analyze for profanity
        analysis = detect_profanity(transcript, threshold=0.5)
        
        # Check if profanity was detected
        if analysis.get("profanity", False):
            profane_words = ", ".join(analysis.get("profane_words", []))
            stream_results["profanity_info"] = f"Profanity Detected (Score: {analysis['score']:.2f})\nProfane Words: {profane_words}"
            
            # Rephrase to clean text
            clean_text = rephrase_profanity(transcript)
            stream_results["clean_text"] = clean_text
            
            # Create audio from cleaned text
            audio_file = text_to_speech(clean_text)
            if audio_file:
                stream_results["audio_output"] = audio_file
        else:
            stream_results["profanity_info"] = f"No Profanity Detected (Score: {analysis['score']:.2f})"
            stream_results["clean_text"] = transcript
            
            # Use original text for audio if no profanity
            audio_file = text_to_speech(transcript)
            if audio_file:
                stream_results["audio_output"] = audio_file
        
        # Clean up temporary file if we created it
        if temp_file and temp_file.startswith("temp_stream_") and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
        
        return stream_results["transcript"], stream_results["profanity_info"], stream_results["clean_text"], stream_results["audio_output"]
        
    except Exception as e:
        error_msg = f"Error processing streaming audio: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        
        # Make sure all models are on CPU if in ZeroGPU mode
        if IS_ZEROGPU and torch.cuda.is_available():
            try:
                whisper_model.to(torch.device("cpu"))
                profanity_model.to(torch.device("cpu"))
                t5_model.to(torch.device("cpu"))
                tts_model.to(torch.device("cpu"))
                vocoder.to(torch.device("cpu"))
            except:
                pass
        
        # Update profanity info with error message
        stream_results["profanity_info"] = f"Error: {str(e)}"
        
        return stream_results["transcript"], stream_results["profanity_info"], stream_results["clean_text"], stream_results["audio_output"]

def start_streaming():
    """Start the real-time audio processing"""
    global processing_active, stream_results
    
    if not models_loaded:
        return "Models not loaded yet. Please wait for initialization to complete."
    
    if processing_active:
        return "Streaming is already active."
    
    # Reset results
    stream_results = {
        "transcript": "",
        "profanity_info": "Waiting for audio input...",
        "clean_text": "",
        "audio_output": None
    }
    
    processing_active = True
    logger.info("Started real-time audio processing")
    return "Started real-time audio processing. Speak into your microphone."

def stop_streaming():
    """Stop the real-time audio processing"""
    global processing_active
    
    if not processing_active:
        return "Streaming is not active."
    
    processing_active = False
    return "Stopped real-time audio processing."

def create_ui():
    """Create the Gradio UI"""
    # Simple CSS for styling
    css = """
    /* Fix for dark mode text visibility */
    .dark .gr-input, 
    .dark textarea,
    .dark .gr-textbox,
    .dark [data-testid="textbox"] {
        color: white !important;
        background-color: #2c303b !important;
    }

    .dark .gr-box, 
    .dark .gr-form, 
    .dark .gr-panel, 
    .dark .gr-block {
        color: white !important;
    }
    
    /* Highlighted text container - with dark mode fixes */
    .highlighted-text {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        background-color: #f9f9f9;
        font-family: sans-serif;
        max-height: 300px;
        overflow-y: auto;
        color: #333 !important; /* Ensure text is dark for light mode */
    }
    
    /* Dark mode specific styling for highlighted text */
    .dark .highlighted-text {
        background-color: #2c303b !important;
        color: #ffffff !important;
        border-color: #4a4f5a !important;
    }
    
    /* Make sure text in the highlighted container remains visible in both themes */
    .highlighted-text, .dark .highlighted-text {
        color-scheme: light dark;
    }
    
    /* Loading animation */
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(0,0,0,.3);
        border-radius: 50%;
        border-top-color: #3498db;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    """

    # Create a custom theme based on Soft but explicitly set to light mode
    light_theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="blue",
        neutral_hue="gray"
    )

    # Set theme to light mode and disable theme switching
    with gr.Blocks(css=css, theme=light_theme, analytics_enabled=False) as ui:
        # Model initialization
        init_status = gr.State("")
        
        gr.Markdown(
            """
            # Profanity Detection & Replacement System
            Detect, rephrase, and listen to cleaned content from text or audio!
            """,
            elem_classes="header"
        )
        
        # The rest of your UI code remains unchanged...
        # Initialize models button with status indicators
        with gr.Row():
            with gr.Column(scale=3):
                init_button = gr.Button("Initialize Models", variant="primary")
                init_output = gr.Textbox(label="Initialization Status", interactive=False)
            with gr.Column(scale=1):
                model_status = gr.HTML(
                    """<div style="text-align: center; padding: 5px;">
                    <p><b>Model Status:</b> <span style="color: #e74c3c;">Not Loaded</span></p>
                    </div>"""
                )
        
        # Global sensitivity slider
        sensitivity = gr.Slider(
            minimum=0.2,
            maximum=0.95,
            value=0.5,
            step=0.05,
            label="Profanity Detection Sensitivity",
            info="Lower values are more permissive, higher values are more strict"
        )

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### Choose an Input Method")

        # Text Analysis
        with gr.Tabs():
            with gr.TabItem("Text Analysis", elem_id="text-tab"):
                with gr.Row():
                    text_input = gr.Textbox(
                        label="Enter Text",
                        placeholder="Type your text here...",
                        lines=5,
                        elem_classes="textbox"
                    )
                with gr.Row():
                    text_button = gr.Button("Analyze Text", variant="primary")
                    clear_button = gr.Button("Clear", variant="secondary")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        text_output = gr.Textbox(label="Results", lines=10)
                        highlighted_output = gr.HTML(label="Detected Profanity", elem_classes="highlighted-text")
                    with gr.Column(scale=1):
                        text_audio_output = gr.Audio(label="Rephrased Audio", type="filepath")

            # Audio Analysis
            with gr.TabItem("Audio Analysis", elem_id="audio-tab"):
                gr.Markdown("### Upload or Record Audio")
                audio_input = gr.Audio(
                    label="Audio Input",
                    type="filepath",
                    sources=["microphone", "upload"]
                    #waveform_options=gr.WaveformOptions(waveform_color="#4a90e2")
                )
                with gr.Row():
                    audio_button = gr.Button("Analyze Audio", variant="primary")
                    clear_audio_button = gr.Button("Clear", variant="secondary")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        audio_output = gr.Textbox(label="Results", lines=10, show_copy_button=True)
                        audio_highlighted_output = gr.HTML(label="Detected Profanity", elem_classes="highlighted-text")
                    with gr.Column(scale=1):
                        clean_audio_output = gr.Audio(label="Rephrased Audio", type="filepath")
            
            # Real-time Streaming
            with gr.TabItem("Real-time Streaming", elem_id="streaming-tab"):
                gr.Markdown("### Real-time Audio Processing")
                gr.Markdown("Enable real-time audio processing to filter profanity as you speak.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        start_stream_button = gr.Button("Start Real-time Processing", variant="primary")
                        stop_stream_button = gr.Button("Stop Real-time Processing", variant="secondary")
                        stream_status = gr.Textbox(label="Streaming Status", value="Inactive", interactive=False)
                        
                        # Add microphone input specifically for streaming
                        stream_audio_input = gr.Audio(
                            label="Streaming Microphone Input",
                            type="filepath",
                            sources=["microphone"],
                            streaming=True
                            #waveform_options=gr.WaveformOptions(waveform_color="#4a90e2")
                        )
                    
                    with gr.Column(scale=2):
                        # Add elements to display streaming results
                        stream_transcript = gr.Textbox(label="Live Transcription", lines=2)
                        stream_profanity_info = gr.Textbox(label="Profanity Detection", lines=2)
                        stream_clean_text = gr.Textbox(label="Clean Text", lines=2)
                        # Element to play the clean audio
                        stream_audio_output = gr.Audio(label="Clean Audio Output", type="filepath")
                
                gr.Markdown("""
                ### How Real-time Streaming Works
                1. Click "Start Real-time Processing" to begin
                2. Use the microphone input to speak
                3. The system will process audio in real-time, detect and clean profanity
                4. You'll see the transcription, profanity info, and clean output appear above
                5. Click "Stop Real-time Processing" when finished
                
                Note: This feature requires microphone access and may have some latency.
                """)

        # Event handlers
        def update_model_status(status_text):
            """Update both the status text and the visual indicator"""
            if "successfully" in status_text.lower():
                status_html = """<div style="text-align: center; padding: 5px;">
                <p><b>Model Status:</b> <span style="color: #2ecc71;">Loaded ✓</span></p>
                </div>"""
            elif "error" in status_text.lower():
                status_html = """<div style="text-align: center; padding: 5px;">
                <p><b>Model Status:</b> <span style="color: #e74c3c;">Error ✗</span></p>
                </div>"""
            else:
                status_html = """<div style="text-align: center; padding: 5px;">
                <p><b>Model Status:</b> <span style="color: #f39c12;">Loading...</span></p>
                </div>"""
            return status_text, status_html
            
        init_button.click(
            lambda: update_model_status("Loading models, please wait..."),
            inputs=[],
            outputs=[init_output, model_status]
        ).then(
            load_models,
            inputs=[],
            outputs=[init_output]
        ).then(
            update_model_status,
            inputs=[init_output],
            outputs=[init_output, model_status]
        )
        
        text_button.click(
            text_analysis, 
            inputs=[text_input, sensitivity], 
            outputs=[text_output, highlighted_output, text_audio_output]
        )
        
        clear_button.click(
            lambda: [None, None, None], 
            inputs=None, 
            outputs=[text_input, highlighted_output, text_audio_output]
        )

        audio_button.click(
            analyze_audio, 
            inputs=[audio_input, sensitivity], 
            outputs=[audio_output, audio_highlighted_output, clean_audio_output]
        )

        clear_audio_button.click(
            lambda: [None, None, None, None], 
            inputs=None, 
            outputs=[audio_input, audio_output, audio_highlighted_output, clean_audio_output]
        )
        
        start_stream_button.click(
            start_streaming,
            inputs=[],
            outputs=[stream_status]
        )
        
        stop_stream_button.click(
            stop_streaming,
            inputs=[],
            outputs=[stream_status]
        )
        
        # Connect the streaming audio input to our processing function
        # First function to debug the audio chunk format
        def debug_audio_format(audio_chunk):
            """Debug function to log audio format"""
            format_info = f"Type: {type(audio_chunk)}"
            if isinstance(audio_chunk, tuple):
                format_info += f", Length: {len(audio_chunk)}"
                for i, item in enumerate(audio_chunk):
                    format_info += f", Item {i} type: {type(item)}"
            logger.info(f"Audio chunk format: {format_info}")
            return audio_chunk
            
        # Use the stream method with preprocessor for debugging
        stream_audio_input.stream(
            fn=process_stream_chunk,
            inputs=[stream_audio_input], 
            outputs=[stream_transcript, stream_profanity_info, stream_clean_text, stream_audio_output],
            preprocess=debug_audio_format
        )
        
    return ui

if __name__ == "__main__":
    # Set environment variable to avoid OpenMP conflicts
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # Create and launch the UI
    ui = create_ui()
    ui.launch(server_name="0.0.0.0", share=True)