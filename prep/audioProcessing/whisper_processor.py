import whisper
import os
import csv
import tempfile
import torch

from moviepy import VideoFileClip
class WhisperProcessor:
    def __init__(self, model_size="large"):
        self.model = whisper.load_model(model_size)

        print("CUDA available:", torch.cuda.is_available())
        print("Whisper model is on device:", self.model.device)

    def transcribe_audio(self, audio_path, language="vi", save_csv_path=None):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file '{audio_path}' not found.")

        # Nếu là video, trích âm thanh ra .wav
        processed_audio_path = self._extract_audio_if_needed(audio_path)

        try:
            result = self.model.transcribe(processed_audio_path, language=language)
            if save_csv_path:
                self._save_to_csv(result["segments"], save_csv_path)
            return result
        finally:
            if processed_audio_path != audio_path and os.path.exists(processed_audio_path):
                os.remove(processed_audio_path)

    def _extract_audio_if_needed(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext in [".wav", ".mp3", ".m4a", ".flac"]:
            return path

        # Tạo file WAV tạm
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

        try:
            print("Đang trích xuất âm thanh từ video bằng moviepy...")
            clip = VideoFileClip(path)
            clip.audio.write_audiofile(temp_wav, fps=16000, codec="pcm_s16le")
            print("Đã trích xuất:", temp_wav)
        except Exception as e:
            print("Lỗi khi dùng moviepy để trích audio:", e)
            raise RuntimeError("Failed to extract audio using moviepy") from e

        return temp_wav

    def _save_to_csv(self, segments, csv_path):
        with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Start Time (s)", "End Time (s)", "Text"])
            for segment in segments:
                writer.writerow([segment["start"], segment["end"], segment["text"]])
