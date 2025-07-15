import sqlite3

class MetadataDB:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.create_tables()

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

    def create_tables(self):
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS Video (
            id TEXT PRIMARY KEY,
            path TEXT,
            name TEXT,
            size INTEGER,
            size_mb REAL,
            format TEXT,
            last_modified TEXT,
            creation_time TEXT
        )
        """)
        print("Tables 'Video' created or already exist.")
        self.conn.commit()

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
        self.conn.close()