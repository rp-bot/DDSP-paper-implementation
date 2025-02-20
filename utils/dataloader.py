import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
import torch.nn.functional as F
import matplotlib.pyplot as plt

test_data = "test_data_annotations.csv"


class NsynthDataset(Dataset):
    def __init__(self, annotation_file, sample_rate=16000, duration=4, transform=None):

        # Load the annotation data from the provided CSV file
        self.df = pd.read_csv(annotation_file)
        self.sample_rate = sample_rate
        self.num_samples = sample_rate * duration

        # Define a default transform if none is provided:
        # Convert waveform to a mel spectrogram then transform amplitude to decibels.
        if transform is None:
            self.transform = torch.nn.Sequential(
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=2048,
                    win_length=2048,
                    hop_length=512,
                    n_mels=128,
                    f_min=20,
                    f_max=8_000,
                ),
                torchaudio.transforms.AmplitudeToDB(),
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Get the file path and label from the dataframe
        row = self.df.iloc[index]
        audio_path, label = row["file_path"], row["instrument"]

        # Load the audio file using torchaudio
        waveform, sr = torchaudio.load(audio_path)

        # Resample the audio if its sample rate doesn't match the desired sample rate
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.sample_rate
            )
            waveform = resampler(waveform)

        # if waveform.shape[1] > self.num_samples:
        #     waveform = waveform[:, : self.num_samples]
        # elif waveform.shape[1] < self.num_samples:
        #     padding = self.num_samples - waveform.shape[1]
        #     waveform = F.pad(waveform, (0, padding))

        # Ensure the audio is 1 channel by taking the mean of all channels
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Transform the waveform into a log mel spectrogram
        spectrogram = self.transform(waveform)

        # match the size of paper

        return spectrogram, label


if __name__ == "__main__":
    dataset = NsynthDataset(test_data)
    spectrogram, label = dataset[1]
    print(spectrogram.shape)
    print(label)
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram[0, :, :].detach().numpy(), cmap="viridis", aspect="auto")
    plt.savefig("temp.png")
