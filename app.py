import streamlit as st
import tempfile
import torch
import soundfile as sf

from transformers import VitsModel, AutoTokenizer

st.title("🔊 AI Text to Speech")

@st.cache_resource
def load_model():
    model_name = "facebook/mms-tts-eng"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = VitsModel.from_pretrained(model_name)

    return tokenizer, model


text = st.text_area(
    "Input the texts",
    placeholder="Example: Hello, I am learning artificial intelligence."
)

if st.button("Generate Speech", type="primary"):

    if not text.strip():
        st.warning("Input your words.")

    else:
        with st.spinner("Loading AI model..."):

            tokenizer, model = load_model()

        inputs = tokenizer(
            text,
            return_tensors="pt"
        )

        with st.spinner("Generating speech..."):

            with torch.no_grad():
                output = model(**inputs).waveform

        audio = output.squeeze().numpy()

        output_path = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ).name

        sf.write(
            output_path,
            audio,
            model.config.sampling_rate
        )

        st.success("Audio generated successfully!")

        st.audio(
            output_path,
            format="audio/wav"
        )
