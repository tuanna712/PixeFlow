import sqlite3

class SQLStore:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def check_table_exists(self, table_name):
        self.cursor.execute("""
        SELECT name FROM sqlite_master WHERE type='table' AND name=?
        """, (table_name,))
        return self.cursor.fetchone() is not None
    
    def get_table_names(self):
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [row[0] for row in self.cursor.fetchall()]
    
    def get_table_info(self, table_name):
        self.cursor.execute(f"PRAGMA table_info({table_name})")
        return self.cursor.fetchall()
    
    def drop_table(self, table_name):
        if self.check_table_exists(table_name):
            self.cursor.execute(f"DROP TABLE {table_name}")
            self.conn.commit()
            print(f"Table '{table_name}' dropped.")
        else:
            print(f"Table '{table_name}' does not exist.")

    # Create TABLES
    def create_video_table(self):
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS Video (
            id TEXT PRIMARY KEY,
            path TEXT,
            name TEXT,
            size INTEGER,
            size_mb REAL,
            format TEXT,
            last_modified TEXT,
            creation_time TEXT,
            processed BOOLEAN DEFAULT 0
        )
        """)
        print("Tables 'Video' created or already exist.")
        self.conn.commit()

    def create_frame_table(self):
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS Frame (
            id TEXT PRIMARY KEY,
            video_id TEXT,
            frame_index INTEGER,
            timestamp REAL,
            bs64 TEXT,
            processed BOOLEAN DEFAULT 0,
            FOREIGN KEY (video_id) REFERENCES Video(id)
        )
        """)
        print("Tables 'Frame' created or already exist.")
        self.conn.commit()

    # Frame CRUD operations
    def insert_frame(self, frame_data):
        self.cursor.execute("""
        INSERT OR REPLACE INTO Frame (id, video_id, frame_index, timestamp, bs64, processed)
        VALUES (?, ?, ?, ?, ?, ?)
        """, frame_data)
        self.conn.commit()

    def fetch_frames(self):
        self.cursor.execute("SELECT * FROM Frame")
        return self.cursor.fetchall()
    
    def fetch_unprocessed_frames(self):
        self.cursor.execute("SELECT id FROM Frame WHERE processed=0")
        return self.cursor.fetchall()
    
    def fetch_processed_frames(self):
        self.cursor.execute("SELECT id FROM Frame WHERE processed=1")
        return self.cursor.fetchall()
    
    def fetch_frame_by_id(self, frame_id):
        self.cursor.execute("SELECT * FROM Frame WHERE id=?", (frame_id,))
        return self.cursor.fetchone()
    
    def fetch_frames_by_video_id(self, video_id):
        self.cursor.execute("SELECT * FROM Frame WHERE video_id=?", (video_id,))
        return self.cursor.fetchall()
    
    def update_frame(self, frame_id, frame_data):
        self.cursor.execute("""
        UPDATE Frame SET video_id=?, frame_index=?, timestamp=?, base64=?
        WHERE id=?
        """, (*frame_data[1:], frame_id))
        self.conn.commit()

    def update_frame_processed(self, frame_id, processed):
        self.cursor.execute("""
        UPDATE Frame SET processed=?
        WHERE id=?
        """, (processed, frame_id))
        self.conn.commit()

    def delete_frame(self, frame_id):
        self.cursor.execute("DELETE FROM Frame WHERE id=?", (frame_id,))
        self.conn.commit()

    def delete_frames_by_video_id(self, video_id):
        self.cursor.execute("DELETE FROM Frame WHERE video_id=?", (video_id,))
        self.conn.commit()

    def delete_all_frames(self):
        self.cursor.execute("DELETE FROM Frame")
        self.conn.commit()

    # Video CRUD operations
    def insert_video(self, video_data):
        self.cursor.execute("""
        INSERT OR REPLACE INTO Video (id, path, name, size, size_mb, format, last_modified, creation_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, video_data)
        self.conn.commit()

    def fetch_videos(self):
        self.cursor.execute("SELECT * FROM Video")
        return self.cursor.fetchall()
    
    def fetch_video_by_path(self, video_path):
        self.cursor.execute("SELECT * FROM Video WHERE path=?", (video_path,))
        return self.cursor.fetchone()
    
    def fetch_video_by_id(self, video_id):
        self.cursor.execute("SELECT * FROM Video WHERE id=?", (video_id,))
        return self.cursor.fetchone()
    
    def update_video(self, video_id, video_data):
        self.cursor.execute("""
        UPDATE Video SET path=?, name=?, size=?, size_mb=?, format=?, last_modified=?, creation_time=?
        WHERE id=?
        """, (*video_data, video_id))
        self.conn.commit()

    def delete_video(self, video_id):
        self.cursor.execute("DELETE FROM Video WHERE id=?", (video_id,))
        self.conn.commit()

    def delete_video_by_path(self, video_path):
        self.cursor.execute("DELETE FROM Video WHERE path=?", (video_path,))
        self.conn.commit()
    
    def delete_all_videos(self):
        self.cursor.execute("DELETE FROM Video")
        self.conn.commit()

    def close(self):
        print("Closing database connection.")
        self.conn.close()