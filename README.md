# 🎙️ Real-Time Conversational Voice Q&A Assistant

[![CI Pipeline](https://github.com/Rohit-bukke/voice-qa-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/Rohit-bukke/voice-qa-assistant/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper-black.svg)](https://github.com/openai/whisper)
[![Ollama](https://img.shields.io/badge/Ollama-Llama_3.2-orange.svg)](https://ollama.com)
[![DSP Filter](https://img.shields.io/badge/DSP-Butterworth_Bandpass-green.svg)](https://github.com/Rohit-bukke/DSP-based-Speech-Noise-Reduction)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A fully offline, privacy-first, hands-free conversational voice AI system. Speak naturally into your microphone, experience continuous **Voice Activity Detection (VAD)** and **DSP-based acoustic noise reduction**, get near-instant transcription via local **OpenAI Whisper**, contextual multi-turn reasoning powered by **LangChain + Ollama (Llama 3.2)**, and sentence-level streaming **Text-to-Speech (TTS)**.

---

## 🌟 Key Features

- 🗣️ **Continuous Hands-Free Mode (Real-Time VAD)**: Dynamic ambient noise calibration with sliding-window RMS energy thresholding and silence cutoff (~1.1s). No manual button presses required.
- 🎛️ **Dual-Stage DSP Preprocessing Pipeline**: Applies a 4th-order Butterworth bandpass filter ($300\text{ Hz} - 3400\text{ Hz}$) and STFT spectral subtraction to suppress background hums and mic hiss prior to STT (integrated from [DSP-based Speech Noise Reduction](https://github.com/Rohit-bukke/DSP-based-Speech-Noise-Reduction)).
- ⚡ **Sentence-Level Streaming TTS Pipeline**: Streams tokens from Llama 3.2 and synthesizes audio at sentence boundaries, lowering perceived Time-To-First-Audio (TTFA) to $< 600\text{ ms}$.
- 🔒 **100% Offline & Privacy-Preserving**: Whisper STT, Ollama LLM, and pyttsx3 TTS run entirely on your local machine. No audio or text is sent to third-party cloud APIs.
- 🧠 **Multi-Turn Sliding-Window Memory & Sessions**: Token-budgeted dialogue state management with automatic turn pruning, disk session persistence, and Markdown export.
- 🌐 **Interactive Web Audio Dashboard**: Sleek dark-mode browser interface featuring real-time FFT frequency spectrum bars, smooth waveform oscillograms, and session management.
- 🧪 **Automated CI/CD Test Suite**: Comprehensive unit tests and GitHub Actions workflow testing DSP filters, memory managers, and API routes.

---

## 🏗️ Architecture & Pipeline

```
                       ┌──────────────────────────────────────────┐
                       │           Live Microphone Input          │
                       └────────────────────┬─────────────────────┘
                                            │
                                            ▼
                       ┌──────────────────────────────────────────┐
                       │     Voice Activity Detection (VAD)       │
                       │   - Dynamic ambient noise calibration    │
                       │   - Real-time sliding-window energy RMS  │
                       └────────────────────┬─────────────────────┘
                                            │
                                            ▼
                       ┌──────────────────────────────────────────┐
                       │          DSP Audio Preprocessor          │
                       │   - DC bias removal & normalization      │
                       │   - Butterworth Bandpass (300-3400Hz)    │
                       │   - STFT Spectral Subtraction            │
                       └────────────────────┬─────────────────────┘
                                            │
                                            ▼
                       ┌──────────────────────────────────────────┐
                       │        OpenAI Whisper STT (Local)        │
                       │        High-accuracy speech-to-text      │
                       └────────────────────┬─────────────────────┘
                                            │
                                            ▼
                       ┌──────────────────────────────────────────┐
                       │     Sliding-Window Memory & Ollama       │
                       │   - Multi-turn context & token budget    │
                       │   - Real-time token streaming (Llama 3.2)│
                       └────────────────────┬─────────────────────┘
                                            │
                                            ▼
                       ┌──────────────────────────────────────────┐
                       │     Sentence-Level Streaming TTS Engine  │
                       │   - Pipelined sentence synthesis         │
                       │   - SHA-256 Audio Response Cache         │
                       └────────────────────┬─────────────────────┘
                                            │
                                            ▼
                                       [ Speaker ]
```

---

## 📂 Project Structure

```
voice-qa-assistant/
├── audio_dsp.py            # Butterworth bandpass filter & STFT spectral subtraction
├── voice_assistant.py      # Main CLI assistant (Continuous VAD + Streaming TTS)
├── memory_manager.py       # Sliding-window memory, token budgeting & session persistence
├── tts_manager.py          # Multi-engine TTS manager with SHA-256 audio cache
├── web_app.py              # FastAPI REST & WebSocket server
├── static/
│   ├── index.html          # Interactive dark-theme Web UI
│   ├── styles.css          # Glassmorphism styling and responsive layout
│   └── app.js              # Dual visualizer (FFT bars + Oscilloscope) & client logic
├── tests/
│   ├── test_dsp.py         # DSP filter and SNR unit tests
│   ├── test_memory.py      # Conversation memory and session tests
│   ├── test_tts.py         # TTS caching and synthesis tests
│   └── run_tests.py        # Standalone test runner
├── scripts/
│   └── benchmark_dsp.py    # Noise reduction & SNR gain benchmarking script
├── .github/workflows/
│   └── ci.yml              # GitHub Actions CI workflow
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore configuration
├── LICENSE                 # MIT License
└── README.md               # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites

- **Python 3.10+**
- **FFmpeg** (Required by Whisper for audio decoding):
  - **Windows**: Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) and add `bin/` to system PATH.
  - **macOS**: `brew install ffmpeg`
  - **Linux**: `sudo apt update && sudo apt install ffmpeg`
- **Ollama**:
  - Download and install [Ollama](https://ollama.com/download).
  - Pull the lightweight, high-performance Llama 3.2 model:
    ```bash
    ollama pull llama3.2
    ```

### 2. Installation

Clone the repository and install the Python dependencies:

```bash
git clone https://github.com/Rohit-bukke/voice-qa-assistant.git
cd voice-qa-assistant
pip install -r requirements.txt
```

---

## 💻 Usage

### Mode A: Continuous Hands-Free Terminal Assistant (Recommended)

Run the assistant in continuous listening mode with real-time sentence-level streaming TTS:

```bash
python voice_assistant.py --mode continuous
```

Options:
- `--mode continuous` : Hands-free voice activation with dynamic VAD.
- `--mode ptt` : Push-to-talk manual mode (Press ENTER to record).
- `--whisper-model base` : Whisper model size (`tiny`, `base`, `small`, `medium`).
- `--model llama3.2` : Ollama model name.
- `--no-dsp` : Disable Butterworth bandpass filtering.
- `--silence-timeout 1.1` : Silence duration threshold in seconds to detect end of speech.

To exit, simply say *"Goodbye"*, *"Exit"*, or *"Stop"*.

---

### Mode B: Modern Web Audio Dashboard (FastAPI)

Launch the interactive web server:

```bash
python web_app.py
```

Then open your browser and navigate to:
```
http://127.0.0.1:8000
```

- Real-time FFT spectrum bars and smooth oscilloscope visualization.
- Multi-session dialogue manager and Markdown export.
- Live latency telemetry (STT, LLM generation, total turnaround).
- Toggle DSP Butterworth filter and STFT spectral subtraction in real time.

---

## 🧪 Testing & Verification

Run the automated test suite locally:

```bash
python tests/run_tests.py
```

Or with `pytest`:
```bash
pytest tests/ -v
```

Run the DSP noise reduction benchmark:
```bash
python scripts/benchmark_dsp.py
```

---

## 🔬 DSP & Noise Reduction Integration

This project directly leverages acoustic filtering principles from our [DSP-based-Speech-Noise-Reduction](https://github.com/Rohit-bukke/DSP-based-Speech-Noise-Reduction) repository:

$$\text{Passband: } 300\text{ Hz} \le f \le 3400\text{ Hz} \quad (\text{4th-Order IIR Butterworth})$$

- **Low-Frequency Suppression**: Filters out electrical AC mains hum ($50/60\text{ Hz}$) and mechanical rumble ($< 300\text{ Hz}$).
- **High-Frequency Suppression**: Filters out ambient microphone hiss, fan noise, and high-frequency harmonics ($> 3400\text{ Hz}$).
- **Zero-Phase Filtering**: Utilizes Second-Order Sections (`sosfiltfilt`) to prevent phase shift and temporal distortion.
- **Spectral Subtraction**: Estimates background noise magnitude spectrum from non-speech frames and attenuates stationary noise floors.

---

## 📜 License

This project is open-source under the [MIT License](LICENSE).
