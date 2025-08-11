import os
import subprocess
import torch
from pathlib import Path
import re
import csv
import sys
from prep.audioProcessing.speech_decoder import SpeechDecoder

class ChunkformerProcessor:
    def __init__(self, repo_dir: str, model_dir: str, device: str = None):
        """
        repo_dir: Thư mục chứa code chunkformer (đã git clone)
        model_dir: Thư mục chứa model chunkformer-large-vie (đã git lfs clone)
        device: 'cuda' hoặc 'cpu'. Nếu None sẽ tự động chọn.
        """
        self.repo_dir = Path(repo_dir)
        self.model_dir = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.decoder = SpeechDecoder(model_dir, device)

        if not self.repo_dir.exists():
            raise FileNotFoundError(f"Repo directory not found: {self.repo_dir}")
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

    def convert_to_wav(self, input_path: str, output_path: str, sample_rate: int = 16000):
        """Dùng ffmpeg để chuyển audio/video sang wav mono"""
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", str(sample_rate),
            "-ac", "1",
            output_path
        ]
        subprocess.run(cmd, check=True)

    def transcribe(self, audio_path: str,
                   total_batch_duration: int = 3600,
                   chunk_size: int = 64,
                   left_context: int = 128,
                   right_context: int = 128):
        """
        Gọi decode.py để nhận transcript.
        """
        current_dir = os.getcwd()
        try:
            os.chdir(self.repo_dir)
            
            # Set PYTHONIOENCODING environment variable
            my_env = os.environ.copy()
            my_env["PYTHONIOENCODING"] = "utf-8"
            
            cmd = [
                sys.executable,
                "decode.py",
                "--model_checkpoint", str(self.model_dir),
                "--long_form_audio", str(Path(audio_path).absolute()),
                "--total_batch_duration", str(total_batch_duration),
                "--chunk_size", str(chunk_size),
                "--left_context_size", str(left_context),
                "--right_context_size", str(right_context)
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                encoding='utf-8',
                env=my_env
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                print("Error output:", stderr)
                raise subprocess.CalledProcessError(process.returncode, cmd)
                
            return stdout
            
        finally:
            os.chdir(current_dir)

    def transcribe_to_csv(self, audio_path: str, csv_path: str,
                          total_batch_duration: int = 3600,
                          chunk_size: int = 64,
                          left_context: int = 128,
                          right_context: int = 128):
        """
        Chạy transcribe và lưu kết quả ra CSV.
        """
        results = self.decoder.decode_audio(
            audio_path,
            chunk_size=chunk_size,
            left_context=left_context,
            right_context=right_context,
            batch_duration=total_batch_duration
        )

        with open(csv_path, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["start_time", "end_time", "text"])
            for item in results:
                writer.writerow([item['start'], item['end'], item['text']])

        print(f"✅ Transcript đã lưu vào {csv_path} ({len(results)} dòng)")