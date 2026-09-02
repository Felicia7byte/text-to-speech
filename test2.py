from transformers import VitsModel, AutoTokenizer
import torch
import soundfile as sf

model_name = "facebook/mms-tts-eng"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = VitsModel.from_pretrained(model_name)

text = "Hello, I am learning artificial intelligence."

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform

audio = output.squeeze().numpy()

sf.write(
    "output.wav",
    audio,
    model.config.sampling_rate
)

print("Audio berhasil dibuat: output.wav")
