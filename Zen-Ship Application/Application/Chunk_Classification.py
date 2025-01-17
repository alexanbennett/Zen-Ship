from VS_Array import get_spectrograms
import soundfile as sf
from keras.models import load_model
from keras import backend as K
import shutil

def f1_score(y_true, y_pred):
    
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


import os
import soundfile as sf

#empty the chunks folders
ex_dir = 'Application/Chunks/Explicit'
none_dir = 'Application/Chunks/Non-Explicit'
if os.path.exists(ex_dir):
    shutil.rmtree(ex_dir)


os.makedirs(ex_dir, exist_ok=True)

if os.path.exists(none_dir):
    shutil.rmtree(none_dir)


os.makedirs(none_dir, exist_ok=True)

def get_chunk_class(chunk_spectrograms): 
    """
    Classifies each audio chunk based on model predection.

    Parameters:
    chunk_spectrograms (array): Chunk and corrosposing spectrogram

    Returns:
    Array: [Chunk, classification].
    """
    model = load_model('CNN\Zen-ShipCNNv2', custom_objects={'f1_score': f1_score})
    chunk_class = []
    excount = 0
    for i, chunktuple in enumerate(chunk_spectrograms):
        currchunk = chunktuple[0]  # This is the audio data
        currspec = chunktuple[1]  # This is the spectrogram
        prediction = model.predict(currspec)
        predicted_class = int(prediction > 0.2)

        # Define the path based on the predicted class
        if predicted_class == 0:
            excount += 1
            folder_path = 'Application/Chunks/Explicit'
        else:
            folder_path = 'Application/Chunks/Non-Explicit'

        # Make sure the directory exists, if not create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Construct the file path where the chunk will be saved
        file_path = os.path.join(folder_path, f'chunk_{i}_pred_{prediction}.wav')

        # Save the chunk to the file
        sf.write(file_path, currchunk, 48000) 

        chunk_class.append((currchunk, predicted_class))

    print("Amount of chunks: ", len(chunk_spectrograms))
    print("Explicit words: ", excount)
    explicit_percentage = 100 * (excount / len(chunk_spectrograms))

    print("Amount of explicit content in audio: ", explicit_percentage)
    return chunk_class
