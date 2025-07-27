import os
import json
import uuid
from datetime import datetime

class Entity:    
    def __init__(self, id):
        self.id = id
        self.dict = {}

    def __str__(self):
        return f"Entity (ID: {self.id})"
    def __repr__(self):
        return f"Entity({self.id})"

class DataDirectory:
    def __init__(self, path):
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv']
        self.path = path
        self.files = self.list_all_files()
        self.videos = []

    def list_all_files(self):
        return [os.path.join(self.path, file) for file in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, file))]
    
    def search_videos(self):
        for file in self.files:
            if any(file.endswith(ext) for ext in self.supported_formats):
                self.videos.append(file)
        return self.videos

class Video(Entity):
    def __init__(self, path, id:str = None):
        id = id or str(uuid.uuid4())
        super().__init__(id)
        self.path = path
        self.name = os.path.basename(path)
        self.metadata = self.get_metadata()

    def get_metadata(self):
        metadata = {}
        if os.path.exists(self.path):
            metadata['path'] = self.path
            metadata['name'] = os.path.basename(self.path)
            metadata['size'] = os.path.getsize(self.path)
            metadata['size_mb'] = round(metadata['size'] / (1024 * 1024), 2)
            metadata['format'] = os.path.splitext(self.path)[1]
            metadata['last_modified'] = datetime.fromtimestamp(os.path.getmtime(self.path)).strftime('%Y-%m-%d %H:%M:%S')
            metadata['creation_time'] = datetime.fromtimestamp(os.path.getctime(self.path)).strftime('%Y-%m-%d %H:%M:%S')
        else:
            print(f"Video '{self.path}' does not exist.")
        return metadata

    
class Frame(Entity):
    def __init__(self, id):
        super().__init__(id)
        self.video_id = None
        self.frame_index = None
        self.timestamp = None
        self.bs64 = None
        self.ocr = None
        self.objects = None
        self.transcription = None
        self.description = None
        self.caption = None
        self.embeddings = None
        
    def to_timestamp_ms(self):
        """
        Convert the timestamp to milliseconds.
        """
        if self.timestamp is not None:
            return int(self.timestamp * 1000)
        return None
    
    def to_timestamp_s(self, timestamp_ms_):
        """
        Convert milliseconds to seconds.
        """
        if timestamp_ms_ is not None:
            return timestamp_ms_ / 1000.0
        return None

class Segment:
    def __init__(self, image, coordinates):
        self.image = image
        self.coordinates = coordinates
        ...


class Audio:
    def __init__(self, path):
        self.path = path
        ...

class Sequence:
    def __init__(self, images):
        self.images = images
        ...