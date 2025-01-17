import librosa
import soundfile as sf
import librosa.display
import matplotlib.pyplot as plt
import torchaudio.transforms as T
import numpy as np
import io
from PIL import Image
import torchaudio
import torch
from tqdm import tqdm
import soundfile as sf

def get_spectrograms(vocals, sr):
    """
    Generates spectrograms for audio chunks derived from a vocal audio signal.

    This function splits an input vocal audio signal into non-silent and silent chunks,
    creates spectrograms for each chunk, and returns a list of tuples containing the
    audio chunks and their corresponding spectrogram arrays.

    Parameters:
    vocals (numpy.ndarray): The vocal audio signal as a 1D NumPy array.
    sr (int): The sample rate of the audio signal.

    Returns:
    array: An array of tuples where each tuple contains:
          - chunk (numpy.ndarray): An audio chunk (silent or non-silent).
          - spectrogram (numpy.ndarray): The corresponding spectrogram array.
    """
    
    #function to create spectrograms, this is applied to each chunk
    def create_spectrogram(audio, sample_rate):
        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format='WAV')
        buf.seek(0)
        waveform, sr = torchaudio.load(buf, format='wav')
        spectrogram = T.Spectrogram(n_fft=400)(waveform)
        plt.figure(figsize=(5, 5))
        plt.axis('off')
        plt.imshow(spectrogram.log2()[0, :, :].detach().numpy(), aspect='auto', origin='lower', cmap='viridis')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        image = Image.open(buf).resize((64, 64))
        img_rgb = image.convert('RGB')
        img_array = np.array(img_rgb).astype('float32')
        img_array /= 255.0
        img_array = np.expand_dims(img_array, axis=0)
        buf.close()

        return img_array

    # Load the audio file
    
    audio_chunks = []
# Save audio data to buffer as WAV
    print("writing sr: ", sr)
    sf.write("vocalsVS_array.wav", vocals, sr, format='WAV')
    # Split the audio signal into non-silent intervals
    vocalsVS_array, samplerate = librosa.load("vocalsVS_array.wav", sr=None)
    print("loading sr: ", samplerate)
    top_db = 30  
    frame_length = 2048  # window size
    hop_length = 512

    # Use librosa.effects.split with the adjusted parameters
    intervals = librosa.effects.split(vocalsVS_array, top_db=top_db, frame_length=frame_length, hop_length=hop_length)

    # Initialize the starting point for the first silent chunk
    last_end = 0
    chunk_counter = 1  # Initialize chunk counter

    for i, interval in enumerate(intervals):
        start_i, end_i = interval

        # Handle silence before the current non-silent chunk
        if start_i > last_end:
            # There's silence before this chunk
            silence_chunk = vocalsVS_array[last_end:start_i]
            audio_chunks.append(silence_chunk)

        # Handle the current non-silent chunk
        current_chunk = vocalsVS_array[start_i:end_i]
        audio_chunks.append(current_chunk)

        # Update last_end to the end of the current chunk
        last_end = end_i

    # Handle silence at the end of the file
    if last_end < len(vocalsVS_array):
        final_silence = vocalsVS_array[last_end:]
        audio_chunks.append(final_silence)



    AuChunk_Spec = []

    for chunk in tqdm(audio_chunks, desc="Creating Spectrograms"):
        spec_array = create_spectrogram(chunk, samplerate)  # Convert chunk to spectrogram
        AuChunk_Spec.append((chunk, spec_array))  # Append the tuple to the list

    return AuChunk_Spec