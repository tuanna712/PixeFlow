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
    def __init__(self, path, video_formats = ['.mp4', '.avi', '.mov', '.mkv']):
        self.video_formats = video_formats
        self.path = path
        self.files = self.list_all_files()
        self.videos = []
        self.non_video_files = []
        self.non_video_files_ext = []

    def list_all_files(self):
        file_list = []
        for root, dirs, files in os.walk(self.path):
            for file in files:
                sub_root = os.path.relpath(root, self.path)
                file_list.append(os.path.join(sub_root, file))
        return file_list
    
    def search_videos(self):
        for file in self.files:
            if any(file.endswith(ext) for ext in self.video_formats):
                self.videos.append(file)
            else:
                self.non_video_files.append(file)
                self.non_video_files_ext.append(os.path.splitext(file)[1])
                self.non_video_files_ext = list(set(self.non_video_files_ext))
        return self.videos

class Video(Entity):
    def __init__(self, path, video_folder_path, id:str = None):
        id = id or str(uuid.uuid4())
        super().__init__(id)
        self.path = path
        self.full_path = os.path.join(video_folder_path, path)
        self.name = os.path.basename(path)
        self.metadata = self.get_metadata()

    def get_metadata(self):
        metadata = {}
        if os.path.exists(self.full_path):
            metadata['path'] = self.path
            metadata['name'] = os.path.basename(self.full_path)
            metadata['size'] = os.path.getsize(self.full_path)
            metadata['size_mb'] = round(metadata['size'] / (1024 * 1024), 2)
            metadata['format'] = os.path.splitext(self.full_path)[1]
            metadata['last_modified'] = datetime.fromtimestamp(os.path.getmtime(self.full_path)).strftime('%Y-%m-%d %H:%M:%S')
            metadata['creation_time'] = datetime.fromtimestamp(os.path.getctime(self.full_path)).strftime('%Y-%m-%d %H:%M:%S')
        else:
            print(f"Video '{self.path}' does not exist.")
        return metadata

    
class Frame(Entity):
    def __init__(self):
        self.id = str(uuid.uuid4())
        super().__init__(self.id)
        self.video_id = None
        self.frame_index = None
        self.frame_path = None
        self.frame_url = None
        
        self.ocr = None
        self.objects = None
        self.transcription = None
        self.description = None
        self.caption = None

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