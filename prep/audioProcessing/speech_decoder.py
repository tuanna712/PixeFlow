import os
import sys
import torch
import torchaudio
import yaml
from pathlib import Path
from typing import List, Dict

# Add chunkformer to Python path
CHUNKFORMER_DIR = Path(__file__).parent / "chunkformer"
sys.path.append(str(CHUNKFORMER_DIR))

# Now imports should work
from model.utils.init_model import init_model
from model.utils.checkpoint import load_checkpoint
from model.utils.file_utils import read_symbol_table
from model.utils.ctc_utils import get_output_with_timestamps
import torchaudio.compliance.kaldi as kaldi
from pydub import AudioSegment

class SpeechDecoder:
    def __init__(self, model_dir: str, device: str = None):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.char_dict = self._init_model()

    def _init_model(self):
        """Khởi tạo model từ checkpoint"""
        config_path = self.model_dir / "config.yaml"
        checkpoint_path = self.model_dir / "pytorch_model.bin"
        symbol_table_path = self.model_dir / "vocab.txt"

        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Initialize model
        model = init_model(config, str(config_path))
        model.eval()
        
        # Load checkpoint
        load_checkpoint(model, str(checkpoint_path))
        
        # Move to device
        model.encoder = model.encoder.to(self.device)
        model.ctc = model.ctc.to(self.device)

        # Load vocabulary
        symbol_table = read_symbol_table(str(symbol_table_path))
        char_dict = {v: k for k, v in symbol_table.items()}

        return model, char_dict

    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file"""
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(16000)
        audio = audio.set_sample_width(2)
        audio = audio.set_channels(1)
        waveform = torch.as_tensor(audio.get_array_of_samples(), dtype=torch.float32).unsqueeze(0)
        return waveform

    @torch.no_grad()
    def decode_audio(self, audio_path: str, 
                    chunk_size: int = 64,
                    left_context: int = 128,
                    right_context: int = 128,
                    batch_duration: int = 3600) -> List[Dict]:
        """
        Decode audio file và trả về list các đoạn text với timestamp
        Returns:
            List[Dict]: List of dicts with keys 'start', 'end', 'text'
        """
        # Load audio
        waveform = self.load_audio(audio_path)
        
        # Extract features
        features = kaldi.fbank(
            waveform,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            energy_floor=0.0,
            sample_frequency=16000
        ).unsqueeze(0)

        # Model config
        subsampling_factor = self.model.encoder.embed.subsampling_factor
        conv_kernel = self.model.encoder.cnn_module_kernel // 2
        max_length = int((batch_duration // 0.01)) // 2

        # Initialize caches
        multiply_n = max_length // chunk_size // subsampling_factor
        truncated_size = chunk_size * multiply_n
        rel_right_size = (right_context + max(chunk_size, right_context) * 
                         (self.model.encoder.num_blocks-1)) * subsampling_factor

        # Process audio in chunks
        hyps = []
        offset = torch.zeros(1, dtype=torch.int, device=self.device)
        att_cache = torch.zeros(
            (self.model.encoder.num_blocks, left_context, 
             self.model.encoder.attention_heads,
             self.model.encoder._output_size * 2 // self.model.encoder.attention_heads)
        ).to(self.device)
        cnn_cache = torch.zeros(
            (self.model.encoder.num_blocks, 
             self.model.encoder._output_size, 
             conv_kernel)
        ).to(self.device)

        # Process chunks
        for idx in range(0, features.shape[1], truncated_size * subsampling_factor):
            start = max(truncated_size * subsampling_factor * idx, 0)
            end = min(truncated_size * subsampling_factor * (idx+1) + 7, features.shape[1])

            x = features[:, start:end+rel_right_size]
            x_len = torch.tensor([x[0].shape[0]], dtype=torch.int).to(self.device)

            # Forward pass
            encoder_out, encoder_len, _, att_cache, cnn_cache, offset = (
                self.model.encoder.forward_parallel_chunk(
                    xs=x,
                    xs_origin_lens=x_len,
                    chunk_size=chunk_size,
                    left_context_size=left_context,
                    right_context_size=right_context,
                    att_cache=att_cache,
                    cnn_cache=cnn_cache,
                    truncated_context_size=truncated_size,
                    offset=offset
                )
            )

            # Process output
            encoder_out = encoder_out.reshape(1, -1, encoder_out.shape[-1])[:, :encoder_len]
            if chunk_size * multiply_n * subsampling_factor * idx + rel_right_size < features.shape[1]:
                encoder_out = encoder_out[:, :truncated_size]

            hyp = self.model.encoder.ctc_forward(encoder_out).squeeze(0)
            hyps.append(hyp)

            if chunk_size * multiply_n * subsampling_factor * idx + rel_right_size >= features.shape[1]:
                break

        # Get final output
        hyps = torch.cat(hyps)
        results = get_output_with_timestamps([hyps], self.char_dict)[0]
        
        # Format results
        output = []
        for item in results:
            output.append({
                'start': item['start'],
                'end': item['end'],
                'text': item['decode']
            })
            
        return output