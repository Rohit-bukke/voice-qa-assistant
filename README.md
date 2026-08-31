# 🎙️ Real-Time Conversational Voice Q&A Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper-black.svg)](https://github.com/openai/whisper)
[![Ollama](https://img.shields.io/badge/Ollama-Llama_3.2-orange.svg)](https://ollama.com)
[![DSP Filter](https://img.shields.io/badge/DSP-Butterworth_Bandpass-green.svg)](https://github.com/Rohit-bukke/DSP-based-Speech-Noise-Reduction)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A fully offline, privacy-first, hands-free conversational voice AI system. Speak naturally into your microphone, experience continuous **Voice Activity Detection (VAD)** and **DSP-based audio noise reduction**, get near-instant transcription via local **OpenAI Whisper**, contextual multi-turn reasoning powered by **LangChain + Ollama (Llama 3.2)**, and clear spoken responses via text-to-speech.

---

## 🌟 Key Features

- 🗣️ **Continuous Hands-Free Mode (Real-Time VAD)**: Dynamic ambient noise calibration with energy thresholding and silence cutoff (~1.1s). No manual button presses required.
- 🎛️ **Integrated DSP Preprocessing Pipeline**: Applies a 4th-order Butterworth bandpass filter ($300\text{ Hz} - 3400\text{ Hz}$) and DC offset removal to suppress ambient noise and isolate vocal formants prior to STT (integrated from [DSP-based Speech Noise Reduction](https://github.com/Rohit-bukke/DSP-based-Speech-Noise-Reduction)).
- 🔒 **100% Offline & Privacy-Preserving**: Whisper STT, Ollama LLM, and pyttsx3 TTS run entirely on your local machine. No audio or text is sent to third-party cloud APIs.
- 🧠 **Multi-Turn Conversational Memory**: LangChain dialogue state tracking retains conversation history across turns for coherent multi-hop discussions.
- 🌐 **Interactive FastAPI Web UI**: Sleek dark-mode browser interface featuring real-time audio waveform visualizers, live transcription bubbles, and audio playback.
- ⚡ **Push-to-Talk (PTT) Support**: Optional manual trigger mode for noisy environments.

---

## 🏗️ Architecture & Pipeline

```
                       ┌──────────────────────────────────────┐
                       │          Live Microphone In          │
                       └──────────────────┬───────────────────┘
                                          │
                                          ▼
                       ┌──────────────────────────────────────┐
                       │   Voice Activity Detection (VAD)     │
                       │   - Dynamic RMS noise calibration    │
                       │   - Automatic speech onset/cutoff    │
                       └──────────────────┬───────────────────┘
                                          │
                                          ▼
                       ┌──────────────────────────────────────┐
                       │        DSP Audio Preprocessor        │
                       │   - DC bias removal & normalization  │
                       │   - Butterworth Bandpass (300-3400Hz)│
                       └──────────────────┬───────────────────┘
                                          │
                                          ▼
                       ┌──────────────────────────────────────┐
                       │      OpenAI Whisper STT (Local)      │
                       │      High-accuracy speech-to-text    │
                       └──────────────────┬───────────────────┘
                                          │
                                          ▼
                       ┌──────────────────────────────────────┐
                       │  LangChain + Ollama (Llama 3.2)      │
                       │  Multi-turn contextual reasoning     │
                       └──────────────────┬───────────────────┘
                                          │
                                          ▼
                       ┌──────────────────────────────────────┐
                       │   Offline TTS (pyttsx3 / Browser)    │
                       │       Synthesized Voice Output       │
                       └──────────────────┬───────────────────┘
                                          │
                                          ▼
                                     [ Speaker ]
```

---

## 📂 Project Structure

```
voice-qa-assistant/
├── audio_dsp.py          # Butterworth bandpass filter & DSP preprocessing
├── voice_assistant.py    # Main CLI assistant (Continuous VAD + Push-to-Talk)
├── web_app.py            # FastAPI REST & WebSocket server
├── static/
│   └── index.html        # Interactive dark-theme Web UI with waveform visualizer
├── requirements.txt      # Python dependencies
├── .gitignore            # Git ignore configuration
└── README.md             # Project documentation
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

Run the assistant in continuous listening mode. It will calibrate to your room's ambient noise and automatically detect when you start and stop talking:

```bash
python voice_assistant.py --mode continuous
```

Options:
- `--mode continuous` : Hands-free voice activation.
- `--mode ptt` : Push-to-talk manual mode (Press ENTER to record).
- `--whisper-model base` : Whisper model size (`tiny`, `base`, `small`, `medium`).
- `--model llama3.2` : Ollama model name.
- `--no-dsp` : Disable Butterworth bandpass filtering.
- `--silence-timeout 1.1` : Silence duration threshold in seconds to detect end of speech.

To exit, simply say *"Goodbye"*, *"Exit"*, or *"Stop"*.

---

### Mode B: Modern Web UI (FastAPI)

Launch the interactive web interface:

```bash
python web_app.py
```

Then open your browser and navigate to:
```
http://127.0.0.1:8000
```

- Live waveform audio visualization.
- Real-time audio processing latency benchmarks.
- Toggle DSP filtering and spoken TTS on the fly.
- Reset conversational context anytime.

---

## 🔬 DSP & Noise Reduction Integration

This project directly leverages acoustic filtering principles from our [DSP-based-Speech-Noise-Reduction](https://github.com/Rohit-bukke/DSP-based-Speech-Noise-Reduction) repository:

$$\text{Passband: } 300\text{ Hz} \le f \le 3400\text{ Hz} \quad (\text{4th-Order IIR Butterworth})$$

- **Low-Frequency Suppression**: Filters out electrical AC mains hum ($50/60\text{ Hz}$) and mechanical desk rumble ($< 300\text{ Hz}$).
- **High-Frequency Suppression**: Filters out ambient microphone hiss, fan noise, and high-frequency harmonics ($> 3400\text{ Hz}$).
- **Zero-Phase Forward-Backward Filtering**: Utilizes Second-Order Sections (`sosfiltfilt`) to prevent phase shift and temporal alignment artifacts.

---

## 📊 Performance & Latency Benchmarks

| Component | Engine / Model | Hardware | Avg Latency |
| :--- | :--- | :--- | :--- |
| **VAD + Preprocessing** | Energy RMS + Butterworth SOS | CPU | $< 15\text{ ms}$ |
| **Speech-to-Text** | Whisper `base` | CPU / GPU | $\sim 0.6 - 1.2\text{ s}$ |
| **LLM Inference** | Ollama `llama3.2:3b` | Local CPU/GPU | $\sim 0.5 - 1.0\text{ s}$ |
| **Text-to-Speech** | `pyttsx3` / Web Speech | Local | Instant streaming |

---

## 📜 License

This project is open-source under the [MIT License](LICENSE).
