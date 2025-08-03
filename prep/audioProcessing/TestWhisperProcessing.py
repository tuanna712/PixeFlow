from prep.audioProcessing.whisper_processor import WhisperProcessor

if __name__ == "__main__":
    audio_path = "D:/AIHCM/PixeFlow/data/vtv24.mp4"
    csv_output = "transcription_output.csv"

    whisperer = WhisperProcessor(model_size="large")
    result = whisperer.transcribe_audio(audio_path, language="vi", save_csv_path=csv_output)
