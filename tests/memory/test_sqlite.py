"""
Tests for SQLiteMemory class.
"""
import os
import tempfile
import pytest
from fence.memory.sqlite import SQLiteMemory
from fence.templates.models import TextContent

def test_add_and_get_messages():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        db_path = tmp.name
    mem = SQLiteMemory(db_path=db_path)
    mem.set_system_message("System initialized.")
    mem.add_user_message("Hello!")
    mem.add_assistant_message("Hi there!")
    messages = mem.get_messages()
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].content[0].text == "Hello!"
    assert messages[1].role == "assistant"
    assert messages[1].content[0].text == "Hi there!"
    assert mem.get_system_message() == "System initialized."
    mem.close()
    os.remove(db_path)

    def test_add_message_with_meta():
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            db_path = tmp.name
        mem = SQLiteMemory(db_path=db_path)
        meta = {"foo": "bar"}
        mem.add_message("user", "Test message", meta=meta)
        messages = mem.get_messages()
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content[0].text == "Test message"
        mem.close()
        os.remove(db_path)