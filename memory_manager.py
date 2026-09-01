"""
================================================================================
CONVERSATION & SLIDING-WINDOW MEMORY MANAGER
================================================================================
Provides session management, token-budgeted sliding-window memory, and
history persistence for conversational voice agents.
================================================================================
"""

import os
import json
import time
import uuid
from typing import List, Dict, Optional, Any


class ConversationTurn:
    """Represents a single exchange turn between User and Assistant."""
    
    def __init__(self, role: str, content: str, timestamp: Optional[float] = None, latency_sec: float = 0.0):
        self.role = role  # "user" | "assistant" | "system"
        self.content = content
        self.timestamp = timestamp or time.time()
        self.latency_sec = latency_sec

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "latency_sec": self.latency_sec
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationTurn":
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", time.time()),
            latency_sec=data.get("latency_sec", 0.0)
        )


class SlidingWindowMemory:
    """
    Manages multi-turn conversation context with sliding-window capacity
    and token budgeting.
    """

    def __init__(self, max_turns: int = 10, max_tokens: int = 2048, system_prompt: Optional[str] = None):
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or (
            "You are a friendly, intelligent, and concise real-time voice assistant. "
            "Respond naturally in 2-3 concise sentences suitable for spoken audio conversation. "
            "Avoid markdown tables or long formatting."
        )
        self.turns: List[ConversationTurn] = []

    def estimate_tokens(self, text: str) -> int:
        """Heuristic estimation of tokens (~4 chars per token in English)."""
        return max(1, len(text) // 4)

    def add_user_message(self, text: str):
        """Records a user utterance."""
        self.turns.append(ConversationTurn(role="user", content=text))
        self._prune_if_needed()

    def add_ai_message(self, text: str, latency_sec: float = 0.0):
        """Records an assistant response."""
        self.turns.append(ConversationTurn(role="assistant", content=text, latency_sec=latency_sec))
        self._prune_if_needed()

    def _prune_if_needed(self):
        """Prunes oldest dialogue turns if max_turns or max_tokens is exceeded."""
        # Turn-based pruning
        if len(self.turns) > self.max_turns * 2:
            # Keep newest turns
            self.turns = self.turns[-(self.max_turns * 2):]

        # Token-based budget enforcement
        total_tokens = sum(self.estimate_tokens(t.content) for t in self.turns)
        while total_tokens > self.max_tokens and len(self.turns) > 2:
            removed = self.turns.pop(0)
            total_tokens -= self.estimate_tokens(removed.content)

    def get_messages_for_langchain(self) -> List[Any]:
        """
        Formats memory for LangChain ChatOllama message lists.
        Imports LangChain classes lazily to avoid strict import-time dependencies.
        """
        try:
            from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
            messages = [SystemMessage(content=self.system_prompt)]
            for turn in self.turns:
                if turn.role == "user":
                    messages.append(HumanMessage(content=turn.content))
                elif turn.role == "assistant":
                    messages.append(AIMessage(content=turn.content))
            return messages
        except ImportError:
            return []

    def clear(self):
        """Clears conversational turns."""
        self.turns.clear()


class SessionManager:
    """
    Manages persistent dialogue sessions saved to disk in JSON format.
    """

    def __init__(self, storage_dir: str = "sessions"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.active_sessions: Dict[str, SlidingWindowMemory] = {}

    def create_session(self, session_id: Optional[str] = None, system_prompt: Optional[str] = None) -> str:
        sid = session_id or str(uuid.uuid4())[:8]
        self.active_sessions[sid] = SlidingWindowMemory(system_prompt=system_prompt)
        return sid

    def get_session(self, session_id: str) -> SlidingWindowMemory:
        if session_id not in self.active_sessions:
            # Attempt to load from disk
            loaded = self._load_from_disk(session_id)
            if loaded:
                self.active_sessions[session_id] = loaded
            else:
                self.active_sessions[session_id] = SlidingWindowMemory()
        return self.active_sessions[session_id]

    def save_session(self, session_id: str):
        if session_id in self.active_sessions:
            memory = self.active_sessions[session_id]
            file_path = os.path.join(self.storage_dir, f"{session_id}.json")
            data = {
                "session_id": session_id,
                "updated_at": time.time(),
                "system_prompt": memory.system_prompt,
                "turns": [t.to_dict() for t in memory.turns]
            }
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    def _load_from_disk(self, session_id: str) -> Optional[SlidingWindowMemory]:
        file_path = os.path.join(self.storage_dir, f"{session_id}.json")
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            mem = SlidingWindowMemory(system_prompt=data.get("system_prompt"))
            for t_dict in data.get("turns", []):
                mem.turns.append(ConversationTurn.from_dict(t_dict))
            return mem
        except Exception:
            return None

    def list_sessions(self) -> List[Dict[str, Any]]:
        sessions = []
        for fn in os.listdir(self.storage_dir):
            if fn.endswith(".json"):
                sid = fn[:-5]
                fp = os.path.join(self.storage_dir, fn)
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        d = json.load(f)
                    sessions.append({
                        "session_id": sid,
                        "turn_count": len(d.get("turns", [])),
                        "updated_at": d.get("updated_at", 0)
                    })
                except Exception:
                    pass
        return sorted(sessions, key=lambda x: x["updated_at"], reverse=True)

    def export_markdown(self, session_id: str) -> str:
        mem = self.get_session(session_id)
        lines = [
            f"# Voice Assistant Conversation Log ({session_id})",
            f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            ""
        ]
        for t in mem.turns:
            role_label = "👤 User" if t.role == "user" else "🤖 Assistant"
            lines.append(f"### {role_label} ({t.latency_sec:.2f}s)")
            lines.append(f"{t.content}\n")
        return "\n".join(lines)
