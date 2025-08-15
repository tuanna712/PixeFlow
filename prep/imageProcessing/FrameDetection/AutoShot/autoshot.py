import os, sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch, ffmpeg
import numpy as np
from PIL import Image
from tqdm import tqdm
import os, cv2, logging, imagehash
from collections import defaultdict
from supernet import TransNetV2Supernet
from typing import List, Optional, Iterator, Tuple, Dict, Any

def get_frames(video_file_path: str, width: int = 48, height: int = 27) -> np.ndarray:
    """
    Extract frames from video 
    Args:
        video_file_path (str): Path to the video file.
        width (int): Width of the extracted frame. Default is 48
        height (int): Height of the extracted frames. Default is 27
    Returns:
        np.ndarray: Array of video frames
    """
    try:
        out, _ = (
             ffmpeg
            .input(video_file_path)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}')
            .run(capture_stdout=True, capture_stderr=True)
        )
        video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
        return video
    except ffmpeg.Error as e:
        print(f"ffmpeg error: {e.stderr.decode()}")
        raise
    except Exception as e:
        print(f"Error in get_frames: {str(e)}")
        raise

def get_batches(frames: np.ndarray):
    """
    Prepare batches of frames for processing. It's like making a video sandwich.
    
    Args:
        frames (np.ndarray): Array of video frames. Try not to feed it pictures of your ex.
    
    Yields:
        np.ndarray: Batches of frames, because processing all at once would make your computer cry.
    """
    reminder = 50 - len(frames) % 50
    if reminder == 50:
        reminder = 0
    frames = np.concatenate([frames[:1]] * 25 + [frames] + [frames[-1:]] * (reminder + 25), 0)

    
    for i in range(0, len(frames) - 50, 50):
        yield frames[i:i + 100]

class AutoShot:
    """
    A class for automatic shot detection in video using the TransNetV2Supernet model.

    This class provides functionality to detect scene changes (shots) in a video
    using a pre-trained neural network model. It's particularly useful in video
    analysis tasks such as content summarization, scene indexing, or video editing.

    Attributes:
        device (str): The device on which the model will run ('cpu' or 'cuda').
        model (torch.nn.Module): The loaded TransNetV2Supernet model.

    Example:
        >>> shot_detector = AutoShot("path/to/pretrained/model.pth")
        >>> scenes = shot_detector.process_video("path/to/video.mp4")
        >>> print(f"Detected {len(scenes)} scenes in the video.")
    """
    def __init__(
        self,
        pretrained_path: str,
        device: Optional[str] = None
    ):
        """
        Initialize the AutoShot class.

        Args:
            pretrained_path (str): Path to the pretrained model weights file.
            device (Optional[str]): Device to run the model on ('cpu' or 'cuda').
                If None, it will use CUDA if available, else CPU.

        Raises:
            RuntimeError: If the model fails to load.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(pretrained_path=pretrained_path)
        
    
    def load_model(self, pretrained_path: str) -> torch.nn.Module:
        """
        Loading the pretrained model

        Args:
            pretrained_path (str): Path to the pretrained model weights

        Raises:
            FileNotFoundError: The pretrained file is missing
            RuntimeError: If loading model process is not successful

        Returns:
            torch.nn.Module : loaded and configured model
        """
        try:
            model =  TransNetV2Supernet().eval()
            if not os.path.exists(pretrained_path):
                raise FileNotFoundError(f"Can't find the pretrained model path at {pretrained_path}")
            
            print(f"Loading the pretrained model from {pretrained_path}")
            model_dict = model.state_dict()
            pretrained_dict = torch.load(pretrained_path, map_location=self.device, weights_only=True)
            pretrained_dict = {k: v for k, v in pretrained_dict['net'].items() if k in model_dict}
            print(f"Current model has {len(model_dict)} params, Updating {len(pretrained_dict)} params")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            return model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load the model. Did you piss off the AI gods? Error: {str(e)}")
    
    def predict(self, batch: np.ndarray) -> np.ndarray:
        """Make predictions on the batch of frames

        Args:
            batch (np.ndarray): Batch of video frames, in the shape of (height, width, color_channel, frames)
            typically: (27, 48, channels=3, frames = 100)

        Returns:
            np.ndarray: Predictions of the batch
        """
        with torch.no_grad():
            batch = torch.from_numpy(batch.transpose((3, 0, 1, 2))[np.newaxis, ...]) * 1.0
            batch = batch.to(self.device)
            one_hot = self.model(batch)

            if isinstance(one_hot, tuple):
                one_hot = one_hot[0]
            return torch.sigmoid(one_hot[0]).cpu().numpy()
    
    def detect_shots(self, frames: np.ndarray) -> np.ndarray:
        """Detects shot in a video

        Args:
            frames (np.ndarray): Array of video frames, (num_frames, height, width, channels)

        Returns:
            np.ndarray: shot detection predictions for each frame
        """
        predictions= []
        
        for batch in tqdm(get_batches(frames=frames), disable=True): #, desc="Dectecting shots", unit="batch"):
            prediction = self.predict(batch=batch)
            predictions.append(prediction[25:75])
        
        return np.concatenate(predictions, axis=0)[:len(frames)]
    
    @staticmethod 
    def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Convert frame-wise predictions to scene boundaries.

        Args:
            predictions (np.ndarray): Array of frame-wise predictions
            threshold (float, optional): Threshold for considering a frame as a shot boundary. Defaults to 0.5

        Returns:
            np.ndarray: List of scene start and end frame indices
        """
        predictions = (predictions > threshold).astype(np.uint8)
        scenes = []
        t, t_prev, start = -1, 0, 0
        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t
        if t == 0:
            scenes.append([start, i])

        # just fix if all predictions are 1
        if len(scenes) == 0:
            return np.array([[0, len(predictions) - 1]], dtype=np.int32)
        return np.array(scenes, dtype=np.int32)

    def process_video(self, video_path: str) -> List[List[int]]:
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"File not found: {video_path}")

            frames = get_frames(video_file_path=video_path)
            if frames is None or len(frames) == 0:
                raise ValueError(f"No frames extracted from video: {video_path}")
            
            predictions = self.detect_shots(frames = frames)
            scenes = self.predictions_to_scenes(predictions=predictions)

            return scenes.tolist()

        except Exception as e:
            raise RuntimeError(F"Failed to process video: {video_path}. Error: {e}")

class DDTNearDuplicateRemoval:
    def __init__(self, threshold: float = 0.80, hash_size: int = 8, verbose: bool = False):
        """
        Initialization of DDTNearDuplicateRemoval
        Args:
            threshold (float, optional): Similarity threshold to be considered as near-duplicate. Defaults to 0.9.
            hash_size (int, optional): The hash size for perceptual hash function. Defaults to 8.
            verbose (bool, optional): Whether to enable detailed logging. Defaults to False.
        """
        self.threshold = threshold
        self.hash_size = hash_size
        self.hash_function = imagehash.phash
        self.image_hashes = []

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info(f"Initialized DDTNearDuplicateRemoval with threshold: {threshold}, hash_size: {hash_size}")

    def preprocess_image(self, *, frame: np.ndarray) -> Image.Image:
        """convert opencv image to PIL image"""
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.logger.debug(f"Preprocessed image, shape: {frame.shape}")
        return pil_image

    def compute_hashes(self, *, keyframes: List[np.ndarray]) -> None:
        """Index the keyframes using DDT"""
        self.image_hashes = []
        self.logger.info(f"Computing hashes for {len(keyframes)} keyframes")
        for i, frame in enumerate(keyframes):
            pil_image = self.preprocess_image(frame=frame)
            image_hash = self.hash_function(pil_image, hash_size=self.hash_size)
            self.image_hashes.append((i, image_hash))
            if (i + 1) % 100 == 0:
                self.logger.info(f"Computed hash for keyframe {i + 1}")
        self.logger.info(f"Finished computing {len(self.image_hashes)} hashes")

    def find_duplicates(self) -> List[List[int]]:
        """Find near duplicate"""
        duplicates = defaultdict(list)
        self.logger.info(f"Finding duplicates among {len(self.image_hashes)} hashes")
        comparison_count = 0
        duplicate_count = 0
        for i in range(len(self.image_hashes)):
            for j in range(i + 1, len(self.image_hashes)):
                comparison_count += 1
                distance = self.image_hashes[i][1] - self.image_hashes[j][1]
                if distance <= self.hash_size * self.hash_size * (1 - self.threshold):
                    duplicates[self.image_hashes[i][0]].append(self.image_hashes[j][0])
                    duplicates[self.image_hashes[j][0]].append(self.image_hashes[i][0])
                    duplicate_count += 1
            if (i + 1) % 100 == 0:
                self.logger.info(f"Processed {i + 1} hashes, found {duplicate_count} duplicates so far")
        self.logger.info(f"Finished duplicate search. Compared {comparison_count} pairs, found {duplicate_count} duplicates")

        duplicate_groups = []
        processed = set()
        for idx, similar in duplicates.items():
            if idx not in processed:
                group = [idx] + similar
                duplicate_groups.append(sorted(set(group)))
                processed.update(group)

        self.logger.info(f"Found {len(duplicate_groups)} duplicate groups")
        return duplicate_groups

    def remove_near_duplicate(self, keyframes: List[np.ndarray]):
        self.logger.info(f"Starting near-duplicate removal for {len(keyframes)} keyframes")
        self.compute_hashes(keyframes=keyframes)
        duplicate_groups = self.find_duplicates()
        unique_indices = set(range(len(keyframes)))
        removed_count = 0
        for group in duplicate_groups:
            unique_indices -= set(group[1:])  # only keep the first same picture index
            removed_count += len(group) - 1

        unique_indices = sorted(unique_indices)
        self.logger.info(f"Finished near-duplicate removal. Kept {len(unique_indices)} unique keyframes, removed {removed_count} duplicates")
        return unique_indices

class KeyFrameExtractor:
    def __init__(self, keyframe_dir: str, verbose: bool = False):
        """
        Initialize the KeyFrameExtractor.

        Args:
            keyframe_dir (str): Directory to save extracted keyframes.
        """
        self.keyframe_dir = keyframe_dir
        self.duplicate_detector = DDTNearDuplicateRemoval(threshold=0.90, hash_size=8, verbose=verbose)
        os.makedirs(self.keyframe_dir, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def sample_frames_from_shot(self, start: int, end: int, num_samples: int = 4) -> List[int]:
        """
        Sample frame indices from a shot.

        Args:
            start (int): Start frame of the shot.
            end (int): End frame of the shot.
            num_samples (int): Number of samples to take.

        Returns:
            List[int]: List of sampled frame indices.
        """
        return [start + i * (end - start) // (num_samples - 1) for i in range(num_samples)]

    def save_frame(self, frame: np.ndarray, filename: str) -> bool:
        """
        Save a frame to disk.

        Args:
            frame (np.ndarray): Frame to save.
            filename (str): Filename to save the frame as.

        Returns:
            bool: True if save was successful, False otherwise.
        """
        return cv2.imwrite(filename, frame, [int(cv2.IMWRITE_WEBP_QUALITY), 80])

    def extract_keyframes(self, video_path: str, scenes: List[List[int]], output_prefix: str) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Extract keyframes from a video, removing near-duplicates.

        Args:
            video_path (str): Path to the video file.
            scenes (List[List[int]]): List of scenes, each containing start and end frame indices.
            output_prefix (str): Prefix for output filenames.

        Yields:
            Tuple[int, np.ndarray]: Tuple of frame index and frame data for unique frames.
        """
        self.logger.info(f"Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        unique_frames = []
        total_frames = 0
        total_unique_frames = 0
        total_duplicate_groups = 0
        total_removed_frames = 0
        try:
            for scene_idx, (start, end) in enumerate(scenes):
                sample_frames = self.sample_frames_from_shot(start, end)
                for frame_idx in sample_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
                    ret, frame = cap.read()
                    if ret:
                        unique_frames.append((frame_idx, frame))
                        if len(unique_frames) >= 50:
                            unique_indices = self.duplicate_detector.remove_near_duplicate([f for _, f in unique_frames])
                            duplicate_groups = self.duplicate_detector.find_duplicates()
                            total_duplicate_groups += len(duplicate_groups)
                            removed_frames = len(unique_frames) - len(unique_indices)
                            total_removed_frames += removed_frames
                            total_unique_frames += len(unique_indices)
                            for idx in unique_indices:
                                yield unique_frames[idx]
                            unique_frames = []
                    else:
                         self.logger.warning(f"Failed to read frame {frame_idx} for video {output_prefix}")

    
            if unique_frames:
                unique_indices = self.duplicate_detector.remove_near_duplicate([f for _, f in unique_frames])
                duplicate_groups = self.duplicate_detector.find_duplicates()
                total_duplicate_groups += len(duplicate_groups)
                removed_frames = len(unique_frames) - len(unique_indices)
                total_removed_frames += removed_frames
                total_unique_frames += len(unique_indices)
                for idx in unique_indices:
                    yield unique_frames[idx]

        finally:
            cap.release()
            self.logger.info(f"Finished processing {video_path}:")
            self.logger.info(f"Total frames processed: {total_frames}")
            self.logger.info(f"Total unique frames: {total_unique_frames}")
            self.logger.info(f"Total duplicate groups: {total_duplicate_groups}")
            self.logger.info(f"Total frames removed: {total_removed_frames}")

    def save_keyframes(self, video_path: str, scenes: List[List[int]], output_prefix: str) -> List[str]:
        """
        Extract keyframes from a video, remove near-duplicates, and save them to disk.

        Args:
            video_path (str): Path to the video file.
            scenes (List[List[int]]): List of scenes, each containing start and end frame indices.
            output_prefix (str): Prefix for output filenames.

        Returns:
            List[str]: List of saved keyframe filenames.
        """
        video_keyframe_dir = os.path.join(self.keyframe_dir, output_prefix)
        os.makedirs(video_keyframe_dir, exist_ok=True)
        
        saved_frames = []
        for frame_idx, frame in self.extract_keyframes(video_path, scenes, output_prefix):
            keyframe_filename = f"{frame_idx}.webp"
            keyframe_path = os.path.join(video_keyframe_dir, keyframe_filename)
            
            if self.save_frame(frame=frame, filename=keyframe_path):
                saved_frames.append(keyframe_path)
            else:
                print(f"Failed to save frame {frame_idx} for video {output_prefix}")
        self.logger.info(f"Saved {len(saved_frames)} keyframes for {video_path}")
        return saved_frames

def dfs_get_video_paths(input_dir: str) -> Iterator[str]:
    """
    Perform an iterative depth-first search to find video files.

    Args:
        input_dir (str): The root directory to search for videos.

    Yields:
        str: Paths to video files.
    """
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    stack = [input_dir]

    while stack:
        current_dir = stack.pop()
        with os.scandir(current_dir) as entries:
            for entry in entries:
                if entry.is_dir():
                    stack.append(entry.path)
                elif entry.is_file() and entry.name.lower().endswith(video_extensions):
                    yield entry.path

def process_single_video(video_path: str, shot_detector: AutoShot, keyframe_extractor: KeyFrameExtractor) -> Dict[str, Any]:
    """
    Process a single video file.

    Args:
        video_path (str): Path to the video file.
        shot_detector (AutoShot): Instance of AutoShot
        keyframe_extractor (KeyFrameExtractor): Instance of KeyFrameExtractor

    Returns:
        Dict[str, Any]: A dictionary containing processing results.
    """
    result = {"video_path": video_path, "status": "success", "scenes": None, "keyframes": None}

    try:
        scenes = shot_detector.process_video(video_path=video_path)
        result["scenes"] = scenes

        if scenes:
            relative_path = os.path.relpath(video_path, os.path.dirname(video_path))
            keyframes = keyframe_extractor.save_keyframes(video_path, scenes, relative_path[:-4]) # remove file extension
            result["keyframes"] = keyframes
        else:
            logging.warning(f"No scenes detected in video: {video_path}")
    except Exception as e:
        logging.error(f"Error processing video {video_path}: {str(e)}")
        result["status"] = "error"
        result["error_message"] = str(e)

    return result

def process_videos(video_path: str, pretrained_model_path: str, keyframe_dir: str, device) -> Dict[str, Dict[str, Any]]:
    """
    Process all videos in the input directory.

    Args:
        input_dir (str): The root directory containing videos to process.
        pretrained_model_path (str): Path to the pretrained model for AutoShot.
        keyframe_dir (str): Directory to save extracted keyframes.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary of processing results for each video.   
    """
    shot_detector = AutoShot(pretrained_model_path, device)
    keyframe_extractor = KeyFrameExtractor(keyframe_dir)

    results = {}
    video_paths = [video_path]
    total_videos = len(video_paths)

    logging.info(f"Starting to process {total_videos} videos")

    for video_path in tqdm(video_paths, desc="Overall Progress", unit="video"):
        relative_path = os.path.relpath(video_path)
        logging.info(f"Processing: {relative_path}")

        result = process_single_video(video_path, shot_detector, keyframe_extractor)
        results[relative_path] = result

    return results


# import sys
# from pathlib import Path
# sys.path.append(str(Path('__file__').resolve().parents[4]))

# pretrained_model_path = "../../../../data/models/Autoshot/checkpoint/ckpt_0_200_0.pth"
# videos_path = "../../../../data/sample/"
# output_keyframe_dir = "../../../../data/sample/keyframes/"

# def main_process():
#     input_dir = [
#         videos_path,
#     ]
#     keyframe_dir = [
#         output_keyframe_dir,
#     ]   
       
#     for i in range(len(input_dir)):
#         results = process_videos(input_dir[i], pretrained_model_path, keyframe_dir[i])

# if __name__ == "__main__":
#     main_process()
#     print("All videos processed successfully.")