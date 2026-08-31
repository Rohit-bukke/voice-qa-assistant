"""
================================================================================
REAL-TIME VOICE Q&A ASSISTANT
================================================================================
A fully offline, hands-free conversational voice assistant:
- Continuous Voice Activity Detection (VAD) & live turn-taking
- DSP Butterworth Bandpass Preprocessing (300Hz - 3400Hz)
- Local Whisper Speech-to-Text
- Multi-turn Conversational Memory with LangChain + Local Ollama (Llama 3.2)
- Fast Offline Text-to-Speech (pyttsx3)
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
import pyttsx3

try:
    import whisper
except ImportError:
    whisper = None

try:
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
except ImportError:
    ChatOllama = None

from audio_dsp import preprocess_speech_audio


# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
DEFAULT_SAMPLE_RATE = 16000     # 16 kHz audio expected by Whisper
CHUNK_DURATION = 0.03          # 30 ms processing chunk
CHUNK_SIZE = int(DEFAULT_SAMPLE_RATE * CHUNK_DURATION)

DEFAULT_WHISPER_MODEL = "base"
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_TTS_RATE = 175

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
    print("🎙️  Calibrating microphone for ambient noise... (please stay quiet for a second)")
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
    print(f" Ambient noise baseline: {avg_noise:.5f} | Speech threshold: {speech_threshold:.5f}\n")
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
        if status:
            pass
        audio_queue.put(indata.copy())

    recorded_chunks = []
    is_speaking = False
    silence_start_time = None
    speech_start_time = None

    print("🟢 Assistant is listening live... Speak anytime (say 'exit' or 'stop' to quit)")
    
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
                        # Reached required silence after speaking
                        if total_duration >= min_speech_duration:
                            print("  [Finished speaking. Processing utterance...]")
                            break
                        else:
                            # Ignored brief click/pop
                            is_speaking = False
                            recorded_chunks = []
                            silence_start_time = None
                else:
                    # Speech resumed during grace period
                    silence_start_time = None

                # Safety max duration limit
                if total_duration >= max_speech_duration:
                    print("  [Max utterance limit reached. Processing...]")
                    break

    if not recorded_chunks:
        return None

    full_audio = np.concatenate(recorded_chunks, axis=0)
    return full_audio


def record_push_to_talk(duration=5.0, sample_rate=DEFAULT_SAMPLE_RATE):
    """Fallback manual recording mode (fixed-duration push-to-talk)."""
    input("\n>>> Press ENTER when ready to speak...")
    print(f"🎤 Recording for {duration} seconds... Speak now!")
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
        processed_audio = preprocess_speech_audio(audio_data, fs=sample_rate, apply_filter=True)
    else:
        processed_audio = audio_data

    # Step 2: Write temporary WAV for Whisper input
    tmp_path = os.path.join(tempfile.gettempdir(), f"voice_in_{os.getpid()}_{int(time.time()*1000)}.wav")
    try:
        audio_int16 = np.int16(np.clip(processed_audio, -1.0, 1.0) * 32767)
        wav_write(tmp_path, sample_rate, audio_int16)
        
        # Step 3: Run Whisper STT
        result = whisper_model.transcribe(tmp_path, fp16=False, language="en")
        transcription = result.get("text", "").strip()
        return transcription
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# ==============================================================================
# LLM INFERENCE (LANGCHAIN + OLLAMA WITH MEMORY)
# ==============================================================================
def initialize_llm(model_name=DEFAULT_OLLAMA_MODEL):
    """Initializes ChatOllama integration."""
    if ChatOllama is None:
        raise ImportError("langchain-ollama is not installed. Please run: pip install langchain-ollama")
    return ChatOllama(model=model_name, temperature=0.7)


def get_llm_response(llm, conversation_history, user_text):
    """
    Appends the user message to history, requests a short conversational response,
    and stores the assistant reply in history.
    """
    conversation_history.append(HumanMessage(content=user_text))
    try:
        response = llm.invoke(conversation_history)
        reply = response.content.strip()
        conversation_history.append(AIMessage(content=reply))
        return reply
    except Exception as e:
        error_msg = f"Sorry, I encountered an error with Ollama: {e}"
        print(f"⚠️  LLM Error: {e}")
        return error_msg


# ==============================================================================
# TEXT-TO-SPEECH (TTS)
# ==============================================================================
def initialize_tts(speech_rate=DEFAULT_TTS_RATE):
    """Initializes offline pyttsx3 engine."""
    engine = pyttsx3.init()
    engine.setProperty("rate", speech_rate)
    return engine


def speak_reply(tts_engine, text):
    """Speaks the response out loud synchronously."""
    if not text:
        return
    tts_engine.say(text)
    tts_engine.runAndWait()


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
    print(" 🎙️  REAL-TIME CONVERSATIONAL VOICE Q&A ASSISTANT")
    print("=" * 70)
    print(f" Mode            : {'Hands-Free Continuous (VAD)' if args.mode == 'continuous' else 'Push-to-Talk (Manual)'}")
    print(f" Whisper STT     : '{args.whisper_model}' (Local)")
    print(f" Ollama LLM      : '{args.model}' (Local)")
    print(f" DSP Bandpass    : {'Enabled (300Hz-3400Hz Butterworth)' if not args.no_dsp else 'Disabled'}")
    print("=" * 70)

    # Check Whisper
    if whisper is None:
        print("❌ Error: 'openai-whisper' package is required. Install with: pip install openai-whisper")
        sys.exit(1)

    print("\n⏳ Loading Whisper model...")
    whisper_model = whisper.load_model(args.whisper_model)

    print("⏳ Connecting to local Ollama LLM...")
    llm = initialize_llm(args.model)

    print("⏳ Initializing Text-to-Speech engine...")
    tts_engine = initialize_tts()

    # System instruction optimized for spoken dialogue
    conversation_history = [
        SystemMessage(content=(
            "You are a friendly, intelligent, and concise real-time voice assistant. "
            "Respond naturally in 2-3 concise sentences suitable for spoken audio conversation. "
            "Avoid markdown tables, markdown formatting, or long lists."
        ))
    ]

    # VAD Noise Calibration
    if args.mode == "continuous":
        threshold = calibrate_ambient_noise()
    else:
        threshold = 0.015

    print("✨ System Ready! Start speaking naturally.\n")

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
            print("📝 Transcribing speech...")
            user_text = transcribe_audio_chunk(
                whisper_model,
                raw_audio,
                sample_rate=DEFAULT_SAMPLE_RATE,
                use_dsp=not args.no_dsp
            )

            if not user_text:
                print("  (No intelligible speech detected)")
                continue

            print(f"\n👤 You: \"{user_text}\"")

            # Check exit phrases
            clean_text = user_text.lower().strip(".,!? ")
            if clean_text in EXIT_PHRASES or any(p in clean_text for p in ["exit assistant", "stop assistant", "goodbye assistant"]):
                print("👋 Assistant: Goodbye! Have a great day.")
                speak_reply(tts_engine, "Goodbye! Have a great day.")
                break

            # 3. LLM Reasoning with Memory
            print("🤔 Assistant thinking...")
            reply = get_llm_response(llm, conversation_history, user_text)
            print(f"🤖 Assistant: \"{reply}\"\n")

            # 4. Speak Reply
            speak_reply(tts_engine, reply)

    except KeyboardInterrupt:
        print("\n\n🛑 Conversation ended by user. Goodbye!")


if __name__ == "__main__":
    main()
