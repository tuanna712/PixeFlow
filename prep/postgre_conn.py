import psycopg2
from psycopg2 import Error

class PostgresDB:
    def __init__(self):
        self.host = "34.63.214.230"
        self.database = "pixelflow"
        self.user = "hcmai25"
        self.password = ""
        self.port = "5432"
        self.connection = None
        self.cursor = None
        self.connect()

    def connect(self):
        """Establishes a connection to the PostgreSQL database."""
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port
            )
            self.cursor = self.connection.cursor()
            print("PostgreSQL connection established.")
        except Error as e:
            print(f"Error connecting to PostgreSQL: {e}")
            self.connection = None
            self.cursor = None

    def get_all_tables(self):
        """Returns a list of all table names in the database."""
        if not self.connection:
            return []
        self.cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        return [row[0] for row in self.cursor.fetchall()]

    def check_table_exists(self, table_name):
        """Checks if a table with the given name exists in the database."""
        if not self.connection:
            print("No connection to the database.")
            return False
        self.cursor.execute("""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_name = %s
            )
        """, (table_name,))
        return self.cursor.fetchone()[0]
    
    def get_table_names(self):
        """Returns a list of all table names in the public schema."""
        if not self.connection:
            return []
        self.cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        return [row[0] for row in self.cursor.fetchall()]
    
    def get_table_info(self, table_name):
        """Returns column information for a given table."""
        if not self.connection:
            return []
        self.cursor.execute(f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = %s
        """, (table_name,))
        return self.cursor.fetchall()
    
    def drop_table(self, table_name):
        """Drops a table if it exists."""
        if not self.connection:
            return
        if self.check_table_exists(table_name):
            self.cursor.execute(f"DROP TABLE {table_name}")
            self.connection.commit()
            print(f"Table '{table_name}' dropped.")
        else:
            print(f"Table '{table_name}' does not exist.")

    # Create TABLES
    def create_video_table(self):
        """Creates the 'Video' table if it doesn't exist."""
        if not self.connection:
            return
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS video (
            id TEXT PRIMARY KEY,
            path TEXT,
            name TEXT,
            size BIGINT,
            size_mb REAL,
            format TEXT,
            last_modified TEXT,
            creation_time TEXT,
            processed BOOLEAN DEFAULT FALSE,
            hold BOOLEAN DEFAULT FALSE,
            hold_by TEXT DEFAULT NULL,
            processed_by TEXT DEFAULT NULL
        )
        """)
        print("Tables 'Video' created or already exist.")
        self.connection.commit()

    def create_frame_table(self):
        """Creates the 'Frame' table if it doesn't exist."""
        if not self.connection:
            return
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS frame (
            id TEXT PRIMARY KEY,
            video_id TEXT,
            frame_index INTEGER,
            frame_path TEXT,
            frame_url TEXT,

            ocr TEXT DEFAULT NULL,
            objects TEXT DEFAULT NULL,
            transcription TEXT DEFAULT NULL,
            description TEXT DEFAULT NULL,
            caption TEXT DEFAULT NULL,

            processed BOOLEAN DEFAULT FALSE,
            hold BOOLEAN DEFAULT FALSE,
            hold_by TEXT DEFAULT NULL,
            processed_by TEXT DEFAULT NULL,
            FOREIGN KEY (video_id) REFERENCES video(id)
        )
        """)
        print("Tables 'Frame' created or already exist.")
        self.connection.commit()

    # Frame CRUD operations
    def insert_frame(self, frame_data):
        """Inserts a new frame or replaces an existing one based on id."""
        if not self.connection:
            print("No database connection.")
            return
        # Check if video_id and frame_index are in database
        def get_frame_indexes_by_video_id(video_id):
            self.cursor.execute("SELECT frame_index FROM frame WHERE video_id=%s", (video_id,))
            return [row[0] for row in self.cursor.fetchall()]
        processed_frame_indexes = get_frame_indexes_by_video_id(frame_data[1])
        if frame_data[2] in processed_frame_indexes:
            print(f"Frame with video_id '{frame_data[1]}' and frame_index '{frame_data[2]}' already exists. Skipping insert.")
        else:
            self.cursor.execute("""
            INSERT INTO frame (id, video_id, frame_index, frame_path, frame_url)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                video_id = EXCLUDED.video_id,
                frame_index = EXCLUDED.frame_index,
                frame_path = EXCLUDED.frame_path,
                frame_url = EXCLUDED.frame_url
            """, frame_data)
            self.connection.commit()

    def fetch_frames(self):
        """Fetches all frames from the database."""
        if not self.connection:
            return []
        self.cursor.execute("SELECT * FROM frame")
        return self.cursor.fetchall()
    
    def fetch_unprocessed_frames(self):
        """Fetches the ids of all unprocessed frames."""
        if not self.connection:
            return []
        self.cursor.execute("SELECT id FROM frame WHERE processed=FALSE")
        return self.cursor.fetchall()
    
    def fetch_processed_frames(self):
        """Fetches the ids of all processed frames."""
        if not self.connection:
            return []
        self.cursor.execute("SELECT id FROM frame WHERE processed=TRUE")
        return self.cursor.fetchall()
    
    def fetch_frame_by_id(self, frame_id):
        """Fetches a single frame by its id."""
        if not self.connection:
            return None
        self.cursor.execute("SELECT * FROM frame WHERE id=%s", (frame_id,))
        return self.cursor.fetchone()
    
    def fetch_frames_by_video_id(self, video_id):
        """Fetches all frames associated with a given video id."""
        if not self.connection:
            return []
        self.cursor.execute("SELECT * FROM frame WHERE video_id=%s", (video_id,))
        return self.cursor.fetchall()
    
    def update_frame(self, frame_id, frame_data):
        """Updates a frame's data."""
        if not self.connection:
            return
        self.cursor.execute("""
        UPDATE frame SET video_id=%s, frame_index=%s, timestamp=%s, bs64=%s
        WHERE id=%s
        """, (*frame_data, frame_id))
        self.connection.commit()

    def update_frame_processed(self, frame_id, processed):
        """Updates a frame's processed status."""
        if not self.connection:
            return
        self.cursor.execute("""
        UPDATE frame SET processed=%s
        WHERE id=%s
        """, (processed, frame_id))
        self.connection.commit()

    def delete_frame(self, frame_id):
        """Deletes a frame by its id."""
        if not self.connection:
            return
        self.cursor.execute("DELETE FROM frame WHERE id=%s", (frame_id,))
        self.connection.commit()

    def delete_frames_by_video_id(self, video_id):
        """Deletes all frames for a specific video."""
        if not self.connection:
            return
        self.cursor.execute("DELETE FROM frame WHERE video_id=%s", (video_id,))
        self.connection.commit()

    def delete_all_frames(self):
        """Deletes all frames from the database."""
        if not self.connection:
            return
        self.cursor.execute("DELETE FROM frame")
        self.connection.commit()

    # Video CRUD operations
    def check_if_video_path_exists(self, video_path):
        """Checks if a video exists in the database."""
        if not self.connection:
            return False
        self.cursor.execute("SELECT EXISTS(SELECT 1 FROM video WHERE path=%s)", (video_path,))
        return self.cursor.fetchone()[0]

    def insert_video(self, video_data):
        """Inserts a new video or replaces an existing one based on id."""
        if not self.connection:
            return
        if self.check_if_video_path_exists(video_data[1]):
            print(f"Video with path '{video_data[1]}' already exists. Skipping insert.")
            return
        self.cursor.execute("""
        INSERT INTO video (id, path, name, size, size_mb, format, last_modified, creation_time)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            path = EXCLUDED.path,
            name = EXCLUDED.name,
            size = EXCLUDED.size,
            size_mb = EXCLUDED.size_mb,
            format = EXCLUDED.format,
            last_modified = EXCLUDED.last_modified,
            creation_time = EXCLUDED.creation_time
        """, video_data)
        self.connection.commit()

    def total_videos(self):
        """Fetches the total number of videos in the database."""
        if not self.connection:
            return 0
        self.cursor.execute("SELECT COUNT(*) FROM video")
        return self.cursor.fetchone()[0]

    def fetch_videos(self):
        """Fetches all videos from the database."""
        if not self.connection:
            return []
        self.cursor.execute("SELECT * FROM video")
        return self.cursor.fetchall()
    
    def fetch_video_by_path(self, video_path):
        """Fetches a single video by its file path."""
        if not self.connection:
            return None
        self.cursor.execute("SELECT * FROM video WHERE path=%s", (video_path,))
        return self.cursor.fetchone()
    
    def fetch_video_by_id(self, video_id):
        """Fetches a single video by its id."""
        if not self.connection:
            return None
        self.cursor.execute("SELECT * FROM video WHERE id=%s", (video_id,))
        return self.cursor.fetchone()
    
    def update_video(self, video_id, video_data):
        """Updates a video's data."""
        if not self.connection:
            return
        self.cursor.execute("""
        UPDATE video SET path=%s, name=%s, size=%s, size_mb=%s, format=%s, last_modified=%s, creation_time=%s
        WHERE id=%s
        """, (*video_data, video_id))
        self.connection.commit()

    def delete_video(self, video_id):
        """Deletes a video by its id."""
        if not self.connection:
            return
        self.cursor.execute("DELETE FROM video WHERE id=%s", (video_id,))
        self.connection.commit()

    def delete_video_by_path(self, video_path):
        """Deletes a video by its file path."""
        if not self.connection:
            return
        self.cursor.execute("DELETE FROM video WHERE path=%s", (video_path,))
        self.connection.commit()

    def delete_all_videos(self):
        """Deletes all videos from the database."""
        if not self.connection:
            return
        self.cursor.execute("DELETE FROM video")
        self.connection.commit()

    def close(self):
        """Closes the database cursor and connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            print("Closing PostgreSQL database connection.")
            self.connection.close()
    
    def __del__(self):
        """Ensures the connection is closed when the object is garbage collected."""
        self.close()

    #=====Custom===Execute=====
    def custom_execute(self, query, params=None):
        """Executes a custom SQL query."""
        if not self.connection:
            return
        self.cursor.execute(query, params)
        self.connection.commit()
        return self.cursor.fetchall() if self.cursor.description else None

    #=====Video===Operation=====
    def get_processed_videos(self):
        """Fetches all processed videos from the database."""
        self.cursor.execute("SELECT * FROM video WHERE processed=TRUE")
        return self.cursor.fetchall()

    def get_unprocessed_videos(self, on_hold=False):
        """Fetches all unprocessed videos from the database."""
        if on_hold:
            self.cursor.execute("SELECT * FROM video WHERE processed=FALSE AND hold=TRUE")
        else:
            self.cursor.execute("SELECT * FROM video WHERE processed=FALSE")
        return self.cursor.fetchall()
    
    def get_videos_on_hold(self, user):
        """Fetches all videos that are on hold."""
        if user:
            self.cursor.execute("SELECT * FROM video WHERE hold=TRUE AND hold_by=%s", (user,))
        else:
            self.cursor.execute("SELECT * FROM video WHERE hold=TRUE")
        return self.cursor.fetchall()

    def get_videos_for_processing(self, hold_by, n_vid=20):
        """
        Fetches n videos that are ready for processing.
        Update hold and hold_by to lock those for processing.
        """
        # Get current hold by user
        self.cursor.execute("SELECT id FROM video WHERE hold=TRUE AND hold_by=%s", (hold_by,))
        current_hold = [row[0] for row in self.cursor.fetchall()]
        if len(current_hold) > 0:
            return current_hold

        # Get unprocessed videos
        self.cursor.execute("SELECT id FROM video WHERE processed=FALSE AND hold=FALSE LIMIT %s",
                            (n_vid,))
        ids = [row[0] for row in self.cursor.fetchall()]
        if len(ids) == 0:
            print("No videos available for processing. All are on hold or processed!")
            return []
        
        # Update video by id
        self.cursor.execute("UPDATE video SET hold=TRUE, hold_by=%s WHERE id IN %s",
                            (hold_by, tuple(ids)))
        self.connection.commit()
        # Return locked ids
        return ids

    def update_processed_video(self, video_id, processed_by):
        """Updates the processed status of videos."""
        if not self.connection:
            return
        # Check if video_id exists
        self.cursor.execute("SELECT EXISTS(SELECT 1 FROM video WHERE id=%s)", (video_id,))
        exists = self.cursor.fetchone()[0]
        if not exists:
            print(f"Video with ID {video_id} does not exist.")
            return
        self.cursor.execute("UPDATE video SET processed=TRUE, hold=FALSE, hold_by=NULL, processed_by=%s WHERE id = %s",
                            (processed_by, video_id))
        print(f"Video {video_id} processed by {processed_by}.")
        self.connection.commit()

#=====Frame===Operation=====
    def get_processed_frames(self):
        """Fetches all processed frames from the database."""
        self.cursor.execute("SELECT * FROM frame WHERE processed=TRUE")
        return self.cursor.fetchall()

    def get_unprocessed_frames(self, on_hold=False):
        """Fetches all unprocessed frames from the database."""
        if on_hold:
            self.cursor.execute("SELECT * FROM frame WHERE processed=FALSE AND hold=TRUE")
        else:
            self.cursor.execute("SELECT * FROM frame WHERE processed=FALSE")
        return self.cursor.fetchall()

    def get_frames_on_hold(self, hold_by):
        """Fetches all frames that are on hold."""
        if hold_by:
            self.cursor.execute("SELECT * FROM frame WHERE hold=TRUE AND hold_by=%s", (hold_by,))
        else:
            self.cursor.execute("SELECT * FROM frame WHERE hold=TRUE")
        return self.cursor.fetchall()

    def get_frames_for_processing(self, hold_by, n_frames=20):
        """
        Fetches n frames that are ready for processing.
        Update hold and hold_by to lock those for processing.
        """
        # Get current hold by user
        self.cursor.execute("SELECT id FROM frame WHERE hold=TRUE AND hold_by=%s", (hold_by,))
        current_hold = [row[0] for row in self.cursor.fetchall()]
        if len(current_hold) > 0:
            return current_hold

        # Get unprocessed frames
        self.cursor.execute("SELECT id FROM frame WHERE processed=FALSE AND hold=FALSE LIMIT %s",
                            (n_frames,))
        ids = [row[0] for row in self.cursor.fetchall()]
        if len(ids) == 0:
            print("No frames available for processing. All are on hold or processed!")
            return []

        # Update frame by id
        self.cursor.execute("UPDATE frame SET hold=TRUE, hold_by=%s WHERE id IN %s",
                            (hold_by, tuple(ids)))
        self.connection.commit()
        # Return locked ids
        return ids

    def update_processed_frame(self, frame_id, processed_by):
        """Updates the processed status of frames."""
        if not self.connection:
            return
        # Check if frame_id exists
        self.cursor.execute("SELECT EXISTS(SELECT 1 FROM frame WHERE id=%s)", (frame_id,))
        exists = self.cursor.fetchone()[0]
        if not exists:
            print(f"Frame with ID {frame_id} does not exist.")
            return
        self.cursor.execute("UPDATE frame SET processed=TRUE, hold=FALSE, hold_by=NULL, processed_by=%s WHERE id = %s",
                            (processed_by, frame_id))
        print(f"Frame {frame_id} processed by {processed_by}.")
        self.connection.commit()
