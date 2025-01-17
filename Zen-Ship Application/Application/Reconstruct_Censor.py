import librosa
import soundfile as sf
import numpy as np

def reconstruct_censor_vocals(chunk_classified, samplerate):
    """
    Reconstructs and censors audio vocals based on classification results.

    Parameters:
    chunk_classified (array): [Audio chunk data, classification label]
    
    samplerate (int): The sample rate of the audio chunks..

    Returns:
    numpy array: The reconstructed and censored audio data.
    """
    recon_vocal_array = []
    # Initialize a variable to keep track of whether the audio is stereo or mono
    is_stereo = len(chunk_classified[0][0].shape) == 2
    
    for chunktuple in chunk_classified:
        chunk = chunktuple[0]
        prediction = chunktuple[1]
        if prediction == 0:
            duration_in_seconds = len(chunk) / samplerate
            # Create a silent chunk with the same number of dimensions as the audio chunks
            if is_stereo:
                silent_chunk = np.zeros((int(duration_in_seconds * samplerate), 2))  # Stereo silent chunk
            else:
                silent_chunk = np.zeros(int(duration_in_seconds * samplerate))  # Mono silent chunk
            recon_vocal_array.append(silent_chunk)
        elif prediction == 1:
            recon_vocal_array.append(chunk)
    
    # Concatenate along the first axis to handle both mono and stereo cases
    censored_vocals = np.concatenate(recon_vocal_array, axis=0)

    return censored_vocals
