"""
================================================================================
FASTAPI REAL-TIME VOICE ASSISTANT SERVER
================================================================================
Provides a modern REST & WebSocket interface for real-time voice conversations
directly in the browser, with audio streaming, DSP filtering, Whisper STT,
LangChain + Ollama inference, and audio synthesis.
================================================================================
"""

import os
import io
import time
import tempfile
import numpy as np
from scipy.io.wavfile import write as wav_write, read as wav_read
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

app = FastAPI(title="Real-Time Voice Q&A Assistant API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models & state
WHISPER_MODEL = None
LLM = None
CONVERSATION_HISTORY = [
    SystemMessage(content=(
        "You are an intelligent, friendly, and concise real-time voice assistant. "
        "Keep your answers short and conversational (2-3 sentences), suited for audio playback."
    ))
]

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def get_whisper():
    global WHISPER_MODEL
    if WHISPER_MODEL is None and whisper is not None:
        print("Loading Whisper model ('base')...")
        WHISPER_MODEL = whisper.load_model("base")
    return WHISPER_MODEL


def get_llm():
    global LLM
    if LLM is None and ChatOllama is not None:
        LLM = ChatOllama(model="llama3.2", temperature=0.7)
    return LLM


class TextQueryRequest(BaseModel):
    query: str


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Real-Time Voice Assistant API is running. Place index.html in static/</h1>")


@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "whisper_available": whisper is not None,
        "langchain_ollama_available": ChatOllama is not None,
        "conversation_turns": len(CONVERSATION_HISTORY) - 1
    }


@app.post("/api/reset")
async def reset_conversation():
    global CONVERSATION_HISTORY
    CONVERSATION_HISTORY = [
        SystemMessage(content=(
            "You are an intelligent, friendly, and concise real-time voice assistant. "
            "Keep your answers short and conversational (2-3 sentences), suited for audio playback."
        ))
    ]
    return {"status": "success", "message": "Conversation history reset."}


@app.post("/api/chat-text")
async def chat_text(payload: TextQueryRequest):
    user_text = payload.query.strip()
    if not user_text:
        return JSONResponse(status_code=400, content={"error": "Empty query"})

    CONVERSATION_HISTORY.append(HumanMessage(content=user_text))
    llm = get_llm()
    
    if llm is not None:
        try:
            response = llm.invoke(CONVERSATION_HISTORY)
            reply = response.content.strip()
        except Exception as e:
            reply = f"Error communicating with local Ollama: {str(e)}"
    else:
        reply = f"Echo response (Ollama offline): I received '{user_text}'"

    CONVERSATION_HISTORY.append(AIMessage(content=reply))
    return {
        "user": user_text,
        "reply": reply,
        "history_length": len(CONVERSATION_HISTORY)
    }


@app.post("/api/process-audio")
async def process_audio(file: UploadFile = File(...), use_dsp: bool = True):
    """
    Receives an uploaded WAV/WebM audio clip from client microphone,
    applies Butterworth DSP noise filtering, transcribes with Whisper,
    generates contextual LLM response, and returns the dialogue.
    """
    start_time = time.time()
    audio_bytes = await file.read()
    
    # Save input audio to temp file
    temp_in = os.path.join(tempfile.gettempdir(), f"upload_{int(time.time()*1000)}.wav")
    with open(temp_in, "wb") as f:
        f.write(audio_bytes)

    try:
        whisper_instance = get_whisper()
        if whisper_instance is None:
            return JSONResponse(status_code=503, content={"error": "Whisper is not initialized or installed"})

        # Load audio data for optional DSP filtering
        try:
            sr, data = wav_read(temp_in)
            if data.dtype == np.int16:
                data_float = data.astype(np.float32) / 32768.0
            else:
                data_float = data.astype(np.float32)
                
            if use_dsp:
                processed = preprocess_speech_audio(data_float, fs=sr, apply_filter=True)
                proc_int16 = np.int16(np.clip(processed, -1.0, 1.0) * 32767)
                wav_write(temp_in, sr, proc_int16)
        except Exception as dsp_err:
            print(f"DSP warning: {dsp_err}")

        # Whisper Transcription
        result = whisper_instance.transcribe(temp_in, fp16=False)
        user_text = result.get("text", "").strip()
        
        if not user_text:
            return {"user": "", "reply": "I couldn't hear any speech clearly. Could you repeat that?", "latency_sec": time.time() - start_time}

        # LLM Response
        CONVERSATION_HISTORY.append(HumanMessage(content=user_text))
        llm = get_llm()
        if llm is not None:
            try:
                response = llm.invoke(CONVERSATION_HISTORY)
                reply = response.content.strip()
            except Exception as e:
                reply = f"Ollama execution error: {e}"
        else:
            reply = f"Transcription: '{user_text}'. (Ensure Ollama is running for dynamic LLM responses)."

        CONVERSATION_HISTORY.append(AIMessage(content=reply))
        latency = round(time.time() - start_time, 2)
        
        return {
            "user": user_text,
            "reply": reply,
            "latency_sec": latency,
            "dsp_applied": use_dsp
        }
    finally:
        if os.path.exists(temp_in):
            try:
                os.remove(temp_in)
            except Exception:
                pass


@app.websocket("/ws/voice-stream")
async def websocket_voice_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time live streaming audio chunks and transcriptions.
    """
    await websocket.accept()
    await websocket.send_json({"status": "connected", "message": "Voice Assistant WebSocket Active"})
    
    try:
        while True:
            data = await websocket.receive_text()
            # Handle client heartbeats or text prompts
            await websocket.send_json({"status": "ack", "received": data})
    except WebSocketDisconnect:
        print("WebSocket client disconnected.")


if __name__ == "__main__":
    import uvicorn
    print("Starting Real-Time Voice Assistant Web Server on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
