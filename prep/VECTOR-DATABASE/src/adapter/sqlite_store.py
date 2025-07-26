import sqlite3
import json
from typing import Any, Optional
from pathlib import Path
import sys
from qdrant_store import QdrantVectorStore

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from models import EmbeddingRecord, VideoRecord, SceneRecord, KeyframeRecord

class SQLiteMetadataStore:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.vector_db = QdrantVectorStore
        self._create_table()
    
    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS embedding_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT,
                image_id TEXT,
                extra_info TEXT,
                FOREIGN KEY (video_id) REFERENCES videos(video_id),
                FOREIGN KEY (image_id) REFERENCES keyframes(keyframe_id)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                video_name TEXT,
                file_name TEXT,
                file_path TEXT,
                file_type TEXT,
                file_date TEXT,
                duration_sec REAL,
                json_path TEXT,
                keyframe_dir TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS scenes (
                scene_id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT,
                start_frame INTEGER,
                end_frame INTEGER,
                FOREIGN KEY (video_id) REFERENCES videos(video_id)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS keyframes (
                keyframe_id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT,
                scene_id INTEGER,
                frame_idx INTEGER,
                file_path TEXT,
                timestamp REAL,
                embedding_id TEXT,
                metadata_json TEXT,
                FOREIGN KEY (video_id) REFERENCES videos(video_id),
                FOREIGN KEY (scene_id) REFERENCES scenes(scene_id)
            )
        """)
        self.conn.commit()

    # ---------- Embedding Metadata CRUD ----------
    def insert(self, record: EmbeddingRecord) -> int:
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO embedding_metadata (video_id, image_id, extra_info)
            VALUES (?, ?, ?)
        """, (record.video_id, record.image_id, json.dumps(record.extra_info)))
        self.conn.commit()
        return cursor.lastrowid
    
    def update(self, record: EmbeddingRecord) -> None:
        self.conn.execute("""
            UPDATE embedding_metadata
            SET video_id = ?, image_id = ?, extra_info = ?
            WHERE id = ?
        """, (record.video_id, record.image_id,
              json.dumps(record.extra_info), record.id))
        self.conn.commit()
    
    def delete(self, record_id: int) -> None:
        self.conn.execute("""
            DELETE FROM embedding_metadata WHERE id = ?
        """, (record_id,))
        self.conn.commit()
        
    def get(self, record_id: int) -> EmbeddingRecord | None:
        row = self.conn.execute("""
            SELECT id, video_id, image_id, extra_info
            FROM embedding_metadata
            WHERE id = ?
        """, (record_id,)).fetchone()
        if row:
            return EmbeddingRecord(
                id=row[0],
                video_id=row[1],
                image_id=row[2],
                embedding=[],
                extra_info=json.loads(row[3])
            )
        return None
    
    # ---------- Video CRUD ----------
    def insert_video(self, record: VideoRecord) -> None:
        self.conn.execute("""
            INSERT OR IGNORE INTO videos VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (record.video_id, record.video_name, record.file_name, record.file_path,
              record.file_type, record.file_date, record.duration_sec, record.json_path, record.keyframe_dir))
        self.conn.commit()

    def update_video(self, record: VideoRecord) -> None:
        self.conn.execute("""
            UPDATE videos SET video_name=?, file_name=?, file_path=?, file_type=?, file_date=?, duration_sec=?, json_path=?, keyframe_dir=?
            WHERE video_id=?
        """, (record.video_name, record.file_name, record.file_path,
              record.file_type, record.file_date, record.duration_sec, record.json_path, record.keyframe_dir, record.video_id))
        self.conn.commit()

    def delete_video(self, video_id: str) -> None:
        self.conn.execute("DELETE FROM videos WHERE video_id=?", (video_id,))
        self.conn.commit()

    def get_video(self, video_id: str) -> Optional[VideoRecord]:
        row = self.conn.execute("SELECT * FROM videos WHERE video_id=?", (video_id,)).fetchone()
        if row:
            return VideoRecord(*row)
        return None

    # ---------- Scene CRUD ----------
    def insert_scene(self, record: SceneRecord) -> int:
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO scenes (video_id, start_frame, end_frame)
            VALUES (?, ?, ?)
        """, (record.video_id, record.start_frame, record.end_frame))
        self.conn.commit()
        return cursor.lastrowid

    def update_scene(self, record: SceneRecord) -> None:
        self.conn.execute("""
            UPDATE scenes SET video_id=?, start_frame=?, end_frame=?
            WHERE scene_id=?
        """, (record.video_id, record.start_frame, record.end_frame, record.scene_id))
        self.conn.commit()

    def delete_scene(self, scene_id: int) -> None:
        self.conn.execute("DELETE FROM scenes WHERE scene_id=?", (scene_id,))
        self.conn.commit()

    def get_scene(self, scene_id: int) -> Optional[SceneRecord]:
        row = self.conn.execute("SELECT * FROM scenes WHERE scene_id=?", (scene_id,)).fetchone()
        if row:
            return SceneRecord(*row)
        return None

    # ---------- Keyframe CRUD ----------
    def insert_keyframe(self, record: KeyframeRecord) -> int:
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO keyframes (video_id, scene_id, frame_idx, file_path, timestamp, embedding_id, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            record.video_id, record.scene_id, record.frame_idx,
            record.file_path, record.timestamp, record.embedding_id,
            json.dumps(record.metadata_json)
        ))
        self.conn.commit()
        return cursor.lastrowid

    def update_keyframe(self, record: KeyframeRecord) -> None:
        self.conn.execute("""
            UPDATE keyframes SET video_id=?, scene_id=?, frame_idx=?, file_path=?, timestamp=?, embedding_id=?, metadata_json=?
            WHERE keyframe_id=?
        """, (
            record.video_id, record.scene_id, record.frame_idx,
            record.file_path, record.timestamp, record.embedding_id,
            json.dumps(record.metadata_json), record.keyframe_id
        ))
        self.conn.commit()

    def delete_keyframe(self, keyframe_id: int) -> None:
        self.conn.execute("DELETE FROM keyframes WHERE keyframe_id=?", (keyframe_id,))
        self.conn.commit()

    def get_keyframe(self, keyframe_id: int) -> Optional[KeyframeRecord]:
        row = self.conn.execute("SELECT * FROM keyframes WHERE keyframe_id=?", (keyframe_id,)).fetchone()
        if row:
            metadata_json = json.loads(row[-1]) if row[-1] else None
            return KeyframeRecord(*row[:-1], metadata_json)
        return None

    def close(self):
        self.conn.close()