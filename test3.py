from TTS.api import TTS

# Load pretrained TTS model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

# Convert text to speech
text = "Hello, I am learning artificial intelligence."

tts.tts_to_file(
    text=text,
    file_path="output.wav"
)

print("Audio berhasil dibuat: output.wav")
