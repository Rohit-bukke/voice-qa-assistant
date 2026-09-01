"""
================================================================================
REAL-TIME VOICE Q&A ASSISTANT
================================================================================
A fully offline, hands-free conversational voice assistant:
- Continuous Voice Activity Detection (VAD) & dynamic ambient calibration
- DSP Butterworth Bandpass Preprocessing (300Hz - 3400Hz) & Normalization
- Local Whisper Speech-to-Text
- Multi-turn Sliding-Window Memory with LangChain + Local Ollama (Llama 3.2)
- Sentence-level Streaming TTS Synthesis Pipeline for ultra-low latency
================================================================================
"""

import argparse
import os
import sys
import time
import queue
import tempfile
import threading
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write

# Ensure UTF-8 stdout on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import whisper
except ImportError:
    whisper = None

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None

from audio_dsp import preprocess_speech_audio
from tts_manager import TTSManager
from memory_manager import SlidingWindowMemory


# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
DEFAULT_SAMPLE_RATE = 16000     # 16 kHz audio expected by Whisper
CHUNK_DURATION = 0.03          # 30 ms processing chunk
CHUNK_SIZE = int(DEFAULT_SAMPLE_RATE * CHUNK_DURATION)

DEFAULT_WHISPER_MODEL = "base"
DEFAULT_OLLAMA_MODEL = "llama3.2"

EXIT_PHRASES = {
    "stop", "exit", "quit", "goodbye", "bye", "terminate",
    "stop listening", "shut down", "good bye"
}


# ==============================================================================
# AUDIO CAPTURE & VOICE ACTIVITY DETECTION (VAD)
# ==============================================================================
def calculate_rms(audio_chunk):
    """Calculates the Root Mean Square (RMS) energy of an audio frame."""
    if len(audio_chunk) == 0:
        return 0.0
    return np.sqrt(np.mean(np.square(audio_chunk), dtype=np.float64))


def calibrate_ambient_noise(sample_rate=DEFAULT_SAMPLE_RATE, duration=1.5):
    """
    Measures the baseline background noise level for dynamic VAD thresholding.
    """
    print("[VAD] Calibrating microphone for ambient noise... (please stay quiet for a moment)")
    num_samples = int(duration * sample_rate)
    recording = sd.rec(num_samples, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    
    rms_values = []
    chunk_len = int(0.05 * sample_rate)
    for i in range(0, len(recording) - chunk_len, chunk_len):
        chunk = recording[i : i + chunk_len]
        rms_values.append(calculate_rms(chunk))
        
    avg_noise = np.mean(rms_values) if rms_values else 0.005
    speech_threshold = max(avg_noise * 2.8, 0.012)
    print(f" Ambient baseline: {avg_noise:.5f} | Speech threshold: {speech_threshold:.5f}\n")
    return speech_threshold


def listen_handsfree_vad(
    sample_rate=DEFAULT_SAMPLE_RATE,
    speech_threshold=0.015,
    silence_duration=1.1,
    min_speech_duration=0.6,
    max_speech_duration=15.0
):
    """
    Continuously monitors the microphone stream. Detects when the user begins
    speaking, captures the audio until silence is detected, and returns the audio array.
    """
    audio_queue = queue.Queue()
    stop_event = threading.Event()

    def audio_callback(indata, frames, time_info, status):
        audio_queue.put(indata.copy())

    recorded_chunks = []
    is_speaking = False
    silence_start_time = None
    speech_start_time = None

    print(" Assistant is listening live... Speak naturally (say 'exit' or 'stop' to quit)")
    
    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SIZE,
        callback=audio_callback
    ):
        while not stop_event.is_set():
            try:
                chunk = audio_queue.get(timeout=0.2).flatten()
            except queue.Empty:
                continue

            rms = calculate_rms(chunk)
            current_time = time.time()

            if not is_speaking:
                if rms > speech_threshold:
                    is_speaking = True
                    speech_start_time = current_time
                    silence_start_time = None
                    recorded_chunks = [chunk]
                    print("  [Speech detected... listening]")
            else:
                recorded_chunks.append(chunk)
                total_duration = current_time - speech_start_time

                # Check if current frame is quiet
                if rms < speech_threshold:
                    if silence_start_time is None:
                        silence_start_time = current_time
                    elif current_time - silence_start_time >= silence_duration:
                        if total_duration >= min_speech_duration:
                            print("  [Finished speaking. Processing speech...]")
                            break
                        else:
                            is_speaking = False
                            recorded_chunks = []
                            silence_start_time = None
                else:
                    silence_start_time = None

                if total_duration >= max_speech_duration:
                    print("  [Max utterance limit reached. Processing...]")
                    break

    if not recorded_chunks:
        return None

    return np.concatenate(recorded_chunks, axis=0)


def record_push_to_talk(duration=5.0, sample_rate=DEFAULT_SAMPLE_RATE):
    """Fallback manual recording mode (push-to-talk)."""
    input("\n>>> Press ENTER when ready to speak...")
    print(f" Recording for {duration} seconds... Speak now!")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


# ==============================================================================
# SPEECH-TO-TEXT (WHISPER) + DSP FILTERING
# ==============================================================================
def transcribe_audio_chunk(whisper_model, audio_data, sample_rate=DEFAULT_SAMPLE_RATE, use_dsp=True):
    """
    Applies DSP bandpass filtering, saves temporary PCM WAV, and transcribes with Whisper.
    """
    if audio_data is None or len(audio_data) == 0:
        return ""

    # Step 1: Apply DSP Butterworth Bandpass & Normalization
    if use_dsp:
        processed_audio = preprocess_speech_audio(audio_data, fs=sample_rate, apply_filter=True, apply_spectral_sub=False)
    else:
        processed_audio = audio_data

    # Step 2: Write temporary WAV for Whisper input
    tmp_path = os.path.join(tempfile.gettempdir(), f"voice_in_{os.getpid()}_{int(time.time()*1000)}.wav")
    try:
        audio_int16 = np.int16(np.clip(processed_audio, -1.0, 1.0) * 32767)
        wav_write(tmp_path, sample_rate, audio_int16)
        
        # Step 3: Run Whisper STT
        result = whisper_model.transcribe(tmp_path, fp16=False, language="en")
        return result.get("text", "").strip()
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# ==============================================================================
# STREAMING LLM & PIPELINED TTS
# ==============================================================================
def stream_llm_and_speak(llm, memory: SlidingWindowMemory, tts: TTSManager, user_text: str) -> str:
    """
    Streams tokens from Ollama. Once sentence delimiters (., !, ?) are encountered,
    dispatches sentence chunks immediately to TTS for ultra-low latency response.
    """
    memory.add_user_message(user_text)
    messages = memory.get_messages_for_langchain()
    
    full_response = []
    sentence_buffer = ""
    sentence_delimiters = {".", "!", "?", "\n"}
    start_time = time.time()

    print(" Assistant: ", end="", flush=True)

    try:
        for chunk in llm.stream(messages):
            token = chunk.content
            print(token, end="", flush=True)
            full_response.append(token)
            sentence_buffer += token

            # Check if sentence boundary formed
            if any(punct in sentence_buffer for punct in sentence_delimiters) and len(sentence_buffer.strip()) > 15:
                # Speak sentence
                tts.speak(sentence_buffer.strip(), block=True)
                sentence_buffer = ""

        # Speak remaining buffer
        if sentence_buffer.strip():
            tts.speak(sentence_buffer.strip(), block=True)

        print()  # newline
        complete_text = "".join(full_response).strip()
        latency = round(time.time() - start_time, 2)
        memory.add_ai_message(complete_text, latency_sec=latency)
        return complete_text

    except Exception as e:
        error_msg = f"Error during Ollama inference: {e}"
        print(f"\n[LLM Error] {e}")
        tts.speak("Sorry, I ran into an issue communicating with the local model.")
        return error_msg


# ==============================================================================
# MAIN CONVERSATIONAL LOOP
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Real-Time Hands-Free Voice Q&A Assistant")
    parser.add_argument(
        "--mode",
        choices=["continuous", "ptt"],
        default="continuous",
        help="Listening mode: 'continuous' (hands-free VAD) or 'ptt' (push-to-talk)."
    )
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL, help="Whisper model size (tiny, base, small, medium)")
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL, help="Ollama local LLM model name (default: llama3.2)")
    parser.add_argument("--no-dsp", action="store_true", help="Disable DSP Butterworth bandpass filtering")
    parser.add_argument("--silence-timeout", type=float, default=1.1, help="Seconds of silence to detect end of speech")
    args = parser.parse_args()

    print("=" * 70)
    print(" [VOICE ASSISTANT] REAL-TIME CONVERSATIONAL SYSTEM")
    print("=" * 70)
    print(f" Mode            : {'Hands-Free Continuous (VAD)' if args.mode == 'continuous' else 'Push-to-Talk (Manual)'}")
    print(f" Whisper STT     : '{args.whisper_model}' (Local)")
    print(f" Ollama LLM      : '{args.model}' (Local)")
    print(f" DSP Bandpass    : {'Enabled (300Hz-3400Hz Butterworth)' if not args.no_dsp else 'Disabled'}")
    print("=" * 70)

    if whisper is None:
        print("[Error] 'openai-whisper' package is required. Install with: pip install openai-whisper")
        sys.exit(1)

    print("\n Loading Whisper model...")
    whisper_model = whisper.load_model(args.whisper_model)

    print(" Connecting to local Ollama LLM...")
    if ChatOllama is not None:
        llm = ChatOllama(model=args.model, temperature=0.7)
    else:
        print("[Error] langchain-ollama is missing. Install with: pip install langchain-ollama")
        sys.exit(1)

    print(" Initializing Text-to-Speech Engine...")
    tts_manager = TTSManager(rate=175)
    memory = SlidingWindowMemory(max_turns=10)

    # VAD Noise Calibration
    if args.mode == "continuous":
        threshold = calibrate_ambient_noise()
    else:
        threshold = 0.015

    print(" System Ready! Start speaking naturally.\n")

    try:
        while True:
            # 1. Capture Audio
            if args.mode == "continuous":
                raw_audio = listen_handsfree_vad(
                    sample_rate=DEFAULT_SAMPLE_RATE,
                    speech_threshold=threshold,
                    silence_duration=args.silence_timeout
                )
            else:
                raw_audio = record_push_to_talk(duration=5.0, sample_rate=DEFAULT_SAMPLE_RATE)

            if raw_audio is None or len(raw_audio) < int(0.4 * DEFAULT_SAMPLE_RATE):
                continue

            # 2. Transcribe with Whisper (+ DSP bandpass)
            print(" Transcribing speech...")
            user_text = transcribe_audio_chunk(
                whisper_model,
                raw_audio,
                sample_rate=DEFAULT_SAMPLE_RATE,
                use_dsp=not args.no_dsp
            )

            if not user_text:
                print("  (No intelligible speech detected)")
                continue

            print(f"\n You: \"{user_text}\"")

            # Check exit phrases
            clean_text = user_text.lower().strip(".,!? ")
            if clean_text in EXIT_PHRASES or any(p in clean_text for p in ["exit assistant", "stop assistant", "goodbye assistant"]):
                print(" Assistant: Goodbye! Have a wonderful day.")
                tts_manager.speak("Goodbye! Have a wonderful day.")
                break

            # 3. Stream LLM & Pipelined TTS
            stream_llm_and_speak(llm, memory, tts_manager, user_text)

    except KeyboardInterrupt:
        print("\n\n Conversation ended. Goodbye!")


if __name__ == "__main__":
    main()
