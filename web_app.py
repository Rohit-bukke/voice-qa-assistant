"""
================================================================================
FASTAPI REAL-TIME VOICE ASSISTANT SERVER
================================================================================
Provides a REST & WebSocket interface for real-time voice conversations
with audio streaming, DSP filtering, Whisper STT, multi-session management,
LangChain + Ollama inference, and audio synthesis.
================================================================================
"""

import os
import time
import tempfile
import numpy as np
from typing import Optional
from scipy.io.wavfile import write as wav_write, read as wav_read
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
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

from audio_dsp import preprocess_speech_audio, compute_snr, compute_spectral_centroid
from memory_manager import SessionManager
from tts_manager import TTSManager

app = FastAPI(
    title="Real-Time Voice Q&A Assistant API",
    description="Offline Voice AI Engine with DSP Bandpass Filtering and Local LLM Inference",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models & managers
WHISPER_MODEL = None
LLM = None
session_manager = SessionManager(storage_dir="sessions")
tts_manager = TTSManager()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def get_whisper():
    global WHISPER_MODEL
    if WHISPER_MODEL is None and whisper is not None:
        print("[API] Loading Whisper model ('base')...")
        WHISPER_MODEL = whisper.load_model("base")
    return WHISPER_MODEL


def get_llm():
    global LLM
    if LLM is None and ChatOllama is not None:
        LLM = ChatOllama(model="llama3.2", temperature=0.7)
    return LLM


class TextQueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"


class CreateSessionRequest(BaseModel):
    session_id: Optional[str] = None
    system_prompt: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Real-Time Voice Assistant API is online.</h1>")


@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "whisper_available": whisper is not None,
        "langchain_ollama_available": ChatOllama is not None,
        "active_sessions_count": len(session_manager.list_sessions()),
        "tts_voices_count": len(tts_manager.get_available_voices())
    }


@app.get("/api/voices")
async def list_voices():
    return {"voices": tts_manager.get_available_voices()}


@app.get("/api/sessions")
async def get_sessions():
    return {"sessions": session_manager.list_sessions()}


@app.post("/api/sessions")
async def create_new_session(payload: CreateSessionRequest):
    sid = session_manager.create_session(
        session_id=payload.session_id,
        system_prompt=payload.system_prompt
    )
    return {"session_id": sid, "status": "created"}


@app.get("/api/sessions/{session_id}")
async def get_session_history(session_id: str):
    memory = session_manager.get_session(session_id)
    return {
        "session_id": session_id,
        "turns": [t.to_dict() for t in memory.turns],
        "system_prompt": memory.system_prompt
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    fp = os.path.join(session_manager.storage_dir, f"{session_id}.json")
    if os.path.exists(fp):
        os.remove(fp)
    if session_id in session_manager.active_sessions:
        del session_manager.active_sessions[session_id]
    return {"status": "deleted", "session_id": session_id}


@app.get("/api/sessions/{session_id}/export")
async def export_session(session_id: str, format: str = Query("markdown", enum=["markdown", "json"])):
    memory = session_manager.get_session(session_id)
    if format == "json":
        return JSONResponse(content={"session_id": session_id, "turns": [t.to_dict() for t in memory.turns]})
    md_content = session_manager.export_markdown(session_id)
    return PlainTextResponse(content=md_content, media_type="text/markdown")


@app.post("/api/chat-text")
async def chat_text(payload: TextQueryRequest):
    user_text = payload.query.strip()
    if not user_text:
        return JSONResponse(status_code=400, content={"error": "Empty query"})

    sid = payload.session_id or "default"
    memory = session_manager.get_session(sid)
    memory.add_user_message(user_text)

    llm = get_llm()
    start_time = time.time()
    if llm is not None:
        try:
            messages = memory.get_messages_for_langchain()
            response = llm.invoke(messages)
            reply = response.content.strip()
        except Exception as e:
            reply = f"Error communicating with local Ollama: {str(e)}"
    else:
        reply = f"Echo response (Ollama offline): I received '{user_text}'"

    latency = round(time.time() - start_time, 2)
    memory.add_ai_message(reply, latency_sec=latency)
    session_manager.save_session(sid)

    return {
        "user": user_text,
        "reply": reply,
        "latency_sec": latency,
        "session_id": sid
    }


@app.post("/api/process-audio")
async def process_audio(
    file: UploadFile = File(...),
    session_id: str = "default",
    use_dsp: bool = True,
    apply_spectral_sub: bool = False
):
    """
    Receives client audio blob, performs DSP filtering (Butterworth + Spectral Subtraction),
    transcribes with Whisper STT, performs contextual inference via Ollama,
    and returns detailed metrics and reply.
    """
    t_start = time.time()
    audio_bytes = await file.read()
    
    temp_in = os.path.join(tempfile.gettempdir(), f"upload_{int(time.time()*1000)}.wav")
    with open(temp_in, "wb") as f:
        f.write(audio_bytes)

    try:
        whisper_instance = get_whisper()
        if whisper_instance is None:
            return JSONResponse(status_code=503, content={"error": "Whisper STT is not initialized"})

        # Load audio data for DSP filtering
        dsp_metrics = {"applied": use_dsp, "spectral_centroid_hz": 0.0}
        try:
            sr, data = wav_read(temp_in)
            if data.dtype == np.int16:
                data_float = data.astype(np.float32) / 32768.0
            else:
                data_float = data.astype(np.float32)
                
            if use_dsp:
                processed = preprocess_speech_audio(
                    data_float,
                    fs=sr,
                    apply_filter=True,
                    apply_spectral_sub=apply_spectral_sub
                )
                dsp_metrics["spectral_centroid_hz"] = round(compute_spectral_centroid(processed, fs=sr), 2)
                proc_int16 = np.int16(np.clip(processed, -1.0, 1.0) * 32767)
                wav_write(temp_in, sr, proc_int16)
        except Exception as dsp_err:
            print(f"[DSP Warning] {dsp_err}")

        # Whisper Transcription
        t_stt_start = time.time()
        result = whisper_instance.transcribe(temp_in, fp16=False)
        stt_latency = round(time.time() - t_stt_start, 2)
        user_text = result.get("text", "").strip()
        
        if not user_text:
            return {
                "user": "",
                "reply": "I couldn't hear any speech clearly. Could you please repeat that?",
                "latency_sec": round(time.time() - t_start, 2),
                "metrics": {"stt_latency_sec": stt_latency, "llm_latency_sec": 0.0}
            }

        # LLM Contextual Inference
        memory = session_manager.get_session(session_id)
        memory.add_user_message(user_text)

        t_llm_start = time.time()
        llm = get_llm()
        if llm is not None:
            try:
                messages = memory.get_messages_for_langchain()
                response = llm.invoke(messages)
                reply = response.content.strip()
            except Exception as e:
                reply = f"Ollama execution error: {e}"
        else:
            reply = f"Transcribed speech: '{user_text}'"

        llm_latency = round(time.time() - t_llm_start, 2)
        total_latency = round(time.time() - t_start, 2)

        memory.add_ai_message(reply, latency_sec=total_latency)
        session_manager.save_session(session_id)
        
        return {
            "user": user_text,
            "reply": reply,
            "latency_sec": total_latency,
            "session_id": session_id,
            "metrics": {
                "stt_latency_sec": stt_latency,
                "llm_latency_sec": llm_latency,
                "dsp": dsp_metrics
            }
        }
    finally:
        if os.path.exists(temp_in):
            try:
                os.remove(temp_in)
            except Exception:
                pass


@app.websocket("/ws/voice-stream")
async def websocket_voice_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({"status": "connected", "message": "Voice Assistant WebSocket Active"})
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({"status": "ack", "received": data})
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    print("[API] Starting Real-Time Voice Assistant Server on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
