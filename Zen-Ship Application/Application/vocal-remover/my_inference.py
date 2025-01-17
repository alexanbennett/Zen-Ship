import argparse
import os

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from lib import dataset
from lib import nets
from lib import spec_utils
from lib import utils


class Separator(object):

    def __init__(self, model, device=None, batchsize=1, cropsize=256, postprocess=False):
        self.model = model
        self.offset = model.offset
        self.device = device
        self.batchsize = batchsize
        self.cropsize = cropsize
        self.postprocess = postprocess

    def _postprocess(self, X_spec, mask):
        if self.postprocess:
            mask_mag = np.abs(mask)
            mask_mag = spec_utils.merge_artifacts(mask_mag)
            mask = mask_mag * np.exp(1.j * np.angle(mask))

        X_mag = np.abs(X_spec)
        X_phase = np.angle(X_spec)

        y_spec = mask * X_mag * np.exp(1.j * X_phase)
        v_spec = (1 - mask) * X_mag * np.exp(1.j * X_phase)
        # y_spec = X_spec * mask
        # v_spec = X_spec - y_spec

        return y_spec, v_spec

    def _separate(self, X_spec_pad, roi_size):
        X_dataset = []
        patches = (X_spec_pad.shape[2] - 2 * self.offset) // roi_size
        for i in range(patches):
            start = i * roi_size
            X_spec_crop = X_spec_pad[:, :, start:start + self.cropsize]
            X_dataset.append(X_spec_crop)

        X_dataset = np.asarray(X_dataset)

        self.model.eval()
        with torch.no_grad():
            mask_list = []
            # To reduce the overhead, dataloader is not used.
            for i in tqdm(range(0, patches, self.batchsize)):
                X_batch = X_dataset[i: i + self.batchsize]
                X_batch = torch.from_numpy(X_batch).to(self.device)

                mask = self.model.predict_mask(torch.abs(X_batch))

                mask = mask.detach().cpu().numpy()
                mask = np.concatenate(mask, axis=2)
                mask_list.append(mask)

            mask = np.concatenate(mask_list, axis=2)

        return mask

    def separate(self, X_spec):
        n_frame = X_spec.shape[2]
        pad_l, pad_r, roi_size = dataset.make_padding(n_frame, self.cropsize, self.offset)
        X_spec_pad = np.pad(X_spec, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_spec_pad /= np.abs(X_spec).max()

        mask = self._separate(X_spec_pad, roi_size)
        mask = mask[:, :, :n_frame]

        y_spec, v_spec = self._postprocess(X_spec, mask)

        return y_spec, v_spec

    def separate_tta(self, X_spec):
        n_frame = X_spec.shape[2]
        pad_l, pad_r, roi_size = dataset.make_padding(n_frame, self.cropsize, self.offset)
        X_spec_pad = np.pad(X_spec, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_spec_pad /= X_spec_pad.max()

        mask = self._separate(X_spec_pad, roi_size)

        pad_l += roi_size // 2
        pad_r += roi_size // 2
        X_spec_pad = np.pad(X_spec, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_spec_pad /= X_spec_pad.max()

        mask_tta = self._separate(X_spec_pad, roi_size)
        mask_tta = mask_tta[:, :, roi_size // 2:]
        mask = (mask[:, :, :n_frame] + mask_tta[:, :, :n_frame]) * 0.5

        y_spec, v_spec = self._postprocess(X_spec, mask)

        return y_spec, v_spec


# ... (other imports and Separator class definition)

def separate_vocals(input_path, sr):
    device = torch.device('cpu')
    pretrained_model = 'Application/vocal-remover/models/baseline.pth'
    n_fft = 2048
    hop_length = 1024
    batchsize = 4
    cropsize = 256
    postprocess = False
    # Assume that the model, utils, and spec_utils modules are already imported and available

    model = nets.CascadedNet(n_fft, hop_length, 32, 128)
    model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
    model.to(device)

    X, sr = librosa.load(input_path, sr=sr, mono=False, dtype=np.float32, res_type='kaiser_fast')
    
    if X.ndim == 1:
        X = np.asarray([X, X])
    
    X_spec = spec_utils.wave_to_spectrogram(X, hop_length, n_fft)
    
    sp = Separator(model=model, device=device, batchsize=batchsize, cropsize=cropsize, postprocess=postprocess)
    y_spec, v_spec = sp.separate(X_spec)
    
    # Convert the separated spectrograms back to waveforms
    vocals_wave = spec_utils.spectrogram_to_wave(y_spec, hop_length=hop_length)
    instrumental_wave = spec_utils.spectrogram_to_wave(v_spec, hop_length=hop_length)

    # Return the waveforms as numpy arrays
    return vocals_wave.T, instrumental_wave.T


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', required=True, help='Input file path')
    args = p.parse_args()
    
    vocals, instrumental = separate_vocals(args.input)
    # You can now save the vocals and instrumental or do further processing



