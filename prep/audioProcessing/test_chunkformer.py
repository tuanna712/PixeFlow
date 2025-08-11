import os
import sys
from pathlib import Path

# Add both project root and chunkformer to Python path
ROOT_DIR = Path(__file__).parent.parent.parent
CHUNKFORMER_DIR = Path(__file__).parent / "chunkformer"
sys.path.extend([str(ROOT_DIR), str(CHUNKFORMER_DIR)])

# Now imports can work
from prep.audioProcessing.chunkformer_processor import ChunkformerProcessor

# Use absolute paths
repo_dir = ROOT_DIR / "prep" / "audioProcessing" / "chunkformer"
model_dir = repo_dir / "chunkformer-large-vie"

try:
    # Initialize processor
    processor = ChunkformerProcessor(str(repo_dir), str(model_dir))

    # Process video
    input_video = ROOT_DIR / "data" / "vtv24.mp4"
    processor.convert_to_wav(str(input_video), "temp_audio.wav")
    processor.transcribe_to_csv("temp_audio.wav", "transcript.csv")

    # Cleanup
    os.remove("temp_audio.wav")
    print("✅ Done!")
except Exception as e:
    print(f"❌ Error: {e}")