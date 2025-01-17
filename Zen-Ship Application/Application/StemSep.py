import sys
sys.path.append('Application/vocal-remover')
from my_inference import separate_vocals
import torchaudio
import torch
import soundfile as sf

#seperate stems
def stemSep(filepath):
    """
    Separates audio into instrumental and vocal stems using a modified vocal remover.

    This function uses a pre-trained vocal remover model from 
    https://github.com/tsurumeso/vocal-remover/releases/tag/v5.1.0 to process an 
    input audio file. It outputs the separated instrumental and vocal components 
    as tensors and saves the vocal stem to a WAV file.

    Parameters:
    filepath (str): Path to the input audio file to be processed.

    Returns:
    tuple: A tuple containing:
           - sample_rate (int): The sample rate of the audio file.
           - instrumental (torch.Tensor): The instrumental stem as a tensor.
           - vocals (torch.Tensor): The vocal stem as a tensor, also saved as 'vocals1.wav'.
    """
    
    waveform, sample_rate = torchaudio.load(filepath, normalize=True)
    instrumental, vocals = separate_vocals(filepath, sample_rate)
    
   
    sf.write('vocals1.wav', vocals, sample_rate)

    return sample_rate, instrumental, vocals
