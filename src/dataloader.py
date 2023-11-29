import os
import pandas as pd
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import Resample
from transformers import ClapProcessor, ClapAudioModelWithProjection


class AudioDataset(Dataset):
    def __init__(self, csv_file, data_dir, model_name, debug):
        """
        :param csv_file:
        :param data_dir:
        :param model_name
        :param debug:
        """
        self.data = pd.read_csv(csv_file)
        self.debug = debug

        if self.debug:
            # To reduce the training duration, select a subset of the training data
            self.data = self.data.sample(frac=0.001)
        self.data_dir = data_dir
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.target_sample_rate = self.processor.feature_extractor.sampling_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_path = os.path.join(self.data_dir, self.data.iloc[idx, 0])
        label = self.data.iloc[idx, 1]

        waveform, original_sample_rate = torchaudio.load(input_path)

        # Resample to model kHz
        resampler = Resample(orig_freq=original_sample_rate, new_freq=self.target_sample_rate)
        resampled_waveform = resampler(waveform.squeeze())

        # Apply model pre-processing
        inputs = self.processor(audios=resampled_waveform, return_tensors="pt", sampling_rate=self.target_sample_rate)
        inputs.data['input_features'] = inputs.data['input_features'].squeeze()
        return inputs, label


if __name__ == '__main__':
    # TODO: convert to unit test
    batch_size = 2
    train_dataset = AudioDataset(csv_file='data/whale-detection-challenge/whale_data/data/custom_train.csv',
                                 data_dir='data/whale-detection-challenge/whale_data/data/train/', model_name="laion/clap-htsat-fused", debug=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    model = ClapAudioModelWithProjection.from_pretrained("laion/clap-htsat-fused")
    for batch_idx, (batch_input, label) in enumerate(train_loader):
        print(batch_input)
        print(label)
        output = model(**batch_input)
        break