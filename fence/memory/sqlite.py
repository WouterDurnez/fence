"""
SQLite-based memory class for agents.
"""

import logging
import sqlite3
from datetime import datetime, UTC
from fence.memory.base import BaseMemory
from fence.templates.models import Content, TextContent, Message

logger = logging.getLogger(__name__)

class SQLiteMemory(BaseMemory):
    """
    SQLite-based memory context for the agent.
    """
    def __init__(self, db_path: str = ":memory:", source: str = None):
        self.db_path = db_path
        self.source = source
        self.conn = sqlite3.connect(self.db_path)
        self._setup_table()
        self.system_message = None

    def _setup_table(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                meta TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def add_message(self, role: str, content: str | Content, meta: dict | None = None):
        if isinstance(content, str):
            content = TextContent(text=content)
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO messages (role, content, meta, timestamp) VALUES (?, ?, ?, ?)",
            (
                role,
                content.text if isinstance(content, TextContent) else str(content),
                str(meta) if meta else None,
                datetime.now(UTC).isoformat()
            )
        )
        self.conn.commit()

    def add_user_message(self, content: str | Content):
        self.add_message(role="user", content=content)

    def add_assistant_message(self, content: str | Content):
        self.add_message(role="assistant", content=content)

    def set_system_message(self, content: str):
        self.system_message = content

    def get_messages(self):
        cur = self.conn.cursor()
        cur.execute("SELECT role, content, meta, timestamp FROM messages ORDER BY id ASC")
        rows = cur.fetchall()
        messages = []
        for row in rows:
            # Create content object with just the text
            content_obj = TextContent(text=row[1])
            messages.append(Message(role=row[0], content=[content_obj]))
        return messages

    def get_system_message(self):
        return self.system_message

    def close(self):
        self.conn.close()
