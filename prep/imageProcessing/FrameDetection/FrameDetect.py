# TransNetV2
import cv2, base64
import uuid
import numpy as np
from PIL import Image
from io import BytesIO
from data.models.TransNetV2.inference.transnetv2 import TransNetV2

from prep.base import Frame

class FrameExtractor(TransNetV2):
    def __init__(self, video_path, video_id):
        """
        Initialize the FrameExtractor with the path to the TransNetV2 model.
        """
        super().__init__()
        self.id = video_id
        self.video_path = video_path
        self.video_frames = None
        self.single_frame_predictions = None
        self.scenes = None
        self.selected_frames = []
        self.frames = []

    def detect_scenes(self):
        self.video_frames, self.single_frame_predictions, self.all_frame_predictions = self.predict_video(self.video_path)
        self.scenes = self.predictions_to_scenes(self.single_frame_predictions)

    def select_frames(self, n_frames=2):
        if n_frames == 2:
            for start, end in self.scenes:
                _range = end - start
                self.selected_frames.append(int(np.ceil(_range / 5) + start))
                self.selected_frames.append(int(np.ceil(end - _range / 5)))
        else:
            for start, end in self.scenes:
                step = (end - start) // n_frames
                self.selected_frames.extend(range(start, end, step))

        self.selected_frames = list(set(self.selected_frames))
        self.selected_frames.sort()

    def frame_to_bs64(self, frame_idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return None
        timestamp = frame_idx / self.fps
        # Convert BGR (OpenCV) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        # Save to in-memory buffer
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        bs64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return bs64, timestamp

    def get_frames(self, k=2):
        self.detect_scenes()
        self.select_frames(n_frames=k)
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        for frame_idx in self.selected_frames:
            bs64, timestamp = self.frame_to_bs64(frame_idx)
            if bs64 is not None:
                frame = Frame(str(uuid.uuid4()))
                frame.video_id = self.id
                frame.frame_index = frame_idx
                frame.timestamp = timestamp
                frame.bs64 = bs64
                self.frames.append(frame)
        self.cap.release()
        return self.frames