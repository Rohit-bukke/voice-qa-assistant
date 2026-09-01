"""
Unit tests for Conversation & Sliding Window Memory Manager.
"""

import os
import sys
import tempfile

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memory_manager import SlidingWindowMemory, SessionManager, ConversationTurn


def test_turn_creation():
    turn = ConversationTurn(role="user", content="Hello assistant", latency_sec=0.45)
    d = turn.to_dict()
    assert d["role"] == "user"
    assert d["content"] == "Hello assistant"
    assert d["latency_sec"] == 0.45

    reconstructed = ConversationTurn.from_dict(d)
    assert reconstructed.role == "user"
    assert reconstructed.content == "Hello assistant"


def test_sliding_window_turn_pruning():
    mem = SlidingWindowMemory(max_turns=3)
    for i in range(5):
        mem.add_user_message(f"User message {i}")
        mem.add_ai_message(f"Assistant reply {i}")

    # Should retain only max_turns * 2 = 6 turns
    assert len(mem.turns) == 6
    assert mem.turns[-1].content == "Assistant reply 4"


def test_sliding_window_token_budgeting():
    mem = SlidingWindowMemory(max_turns=10, max_tokens=20)
    # Adding a very long message
    long_msg = "This is a very long message that contains a lot of words and exceeds the token limit."
    mem.add_user_message(long_msg)
    mem.add_ai_message("Short reply")
    mem.add_user_message("Another message")

    assert len(mem.turns) >= 2


def test_session_manager_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        sm = SessionManager(storage_dir=tmpdir)
        sid = sm.create_session("test_session")
        mem = sm.get_session(sid)
        mem.add_user_message("How is the weather?")
        mem.add_ai_message("It is sunny.")
        sm.save_session(sid)

        # Create a new SessionManager pointing to same directory
        sm2 = SessionManager(storage_dir=tmpdir)
        loaded_mem = sm2.get_session(sid)
        assert len(loaded_mem.turns) == 2
        assert loaded_mem.turns[0].content == "How is the weather?"

        # Test Markdown Export
        md = sm2.export_markdown(sid)
        assert "# Voice Assistant Conversation Log (test_session)" in md
        assert "How is the weather?" in md
