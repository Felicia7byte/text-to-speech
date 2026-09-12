# text-to-speech
An AI-powered text-to-speech application built with Python and Streamlit using the pre-trained Meta MMS English Text-to-Speech model from Hugging Face Transformers.
# Features
- Convert English text into speech
- Generate audio from user-provided text
- Play the generated speech directly in the browser
- Generate audio in WAV format
- Simple and interactive interface built with Streamlit
# Tech Stack
Python, Streamlit, Hugging Face Transformers, PyTorch, SoundFile, Meta MMS TTS
# How It Works
Enter Text → Tokenization → MMS TTS Model → Speech Waveform Generation → WAV Audio → Audio Playback

The input text is first tokenized using the model's tokenizer. The VITS-based model then generates an audio waveform, which is converted into a NumPy array and saved as a WAV file using SoundFile.

# Model
Meta MMS English Text-to-Speech
# Live Model
https://text-to-speech-torch.streamlit.app/
