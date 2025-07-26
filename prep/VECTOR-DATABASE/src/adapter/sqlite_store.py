import sqlite3
import json
from typing import Any
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from models import EmbeddingRecord

class SQLiteMetadataStore:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self._create_table()
    
    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS embedding_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT,
                keyframe_id TEXT,
                extra_info TEXT
            )
        """)
        self.conn.commit()
    
    def insert(self, record: EmbeddingRecord) -> int:
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO embedding_metadata (video_id, keyframe_id, extra_info)
            VALUES (?, ?, ?)
        """, (record.video_id, record.keyframe_id, json.dumps(record.extra_info)))
        self.conn.commit()
        return cursor.lastrowid
    
    def update(self, record: EmbeddingRecord) -> None:
        self.conn.execute("""
            UPDATE embedding_metadata
            SET video_id = ?, keyframe_id = ?, extra_info = ?
            WHERE id = ?
        """, (record.video_id, record.keyframe_id,
              json.dumps(record.extra_info), record.id))
        self.conn.commit()
    
    def delete(self, record_id: int) -> None:
        self.conn.execute("""
            DELETE FROM embedding_metadata WHERE id = ?
        """, (record_id,))
        self.conn.commit()
        
    def get(self, record_id: int) -> EmbeddingRecord | None:
        row = self.conn.execute("""
            SELECT id, video_id, keyframe_id, extra_info
            FROM embedding_metadata
            WHERE id = ?
        """, (record_id,)).fetchone()
        if row:
            return EmbeddingRecord(
                id=row[0],
                video_id=row[1],
                keyframe_id=row[2],
                embeddings={},
                extra_info=json.loads(row[3]),
                score=0.0
            )
        return None